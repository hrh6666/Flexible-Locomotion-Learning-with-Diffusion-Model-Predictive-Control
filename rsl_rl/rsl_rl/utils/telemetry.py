# rsl_rl/utils/helpers.py
import os, time
from typing import Dict, Tuple, Optional
import torch

class DiffusionTelemetry:
    """
    Minimal telemetry buffer that appends *every call* to disk.

    Records per append:
      - x   : full trajectory [B, H, T] (cast to dtype)
      - cmd : cond[pivot][:, cmd_slice] -> [B, 3]
      - t   : reverse step index per row -> [B]

    Modes:
      - Sharded (legacy): accumulate rows then save shards via torch.save
      - Single-file stream: append each record to one file via an open handle
    """
    def __init__(
        self,
        out_dir: str,
        *,
        flush_every_rows: int = 200,                # ignored in single-file mode
        dtype: torch.dtype = torch.float16,
        cmd_slice: Tuple[int, int] = (9, 12),
        tag: str = "",
        single_file_path: Optional[str] = None,     # e.g., "tele_stream.pt"
        overwrite_single_file: bool = True,
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.flush_every_rows = max(1, int(flush_every_rows))
        self.dtype = dtype
        self.cmd_slice = cmd_slice
        self.tag = tag

        # legacy shard buffers
        self._x_list, self._cmd_list, self._t_list = [], [], []
        self._rows, self._shard_idx = 0, 0

        # single-file stream mode
        self._single_file_path = single_file_path
        self._f = None
        if self._single_file_path is not None:
            if not os.path.isabs(self._single_file_path):
                self._single_file_path = os.path.join(self.out_dir, self._single_file_path)
            if overwrite_single_file or (not os.path.exists(self._single_file_path)):
                open(self._single_file_path, "wb").close()
            self._f = open(self._single_file_path, "ab")  # keep open

    @torch.no_grad()
    def maybe_log(self, x: torch.Tensor, cond: Dict[int, torch.Tensor], t: torch.Tensor):
        """
        Append one record *unconditionally* for this reverse step.
        Args:
          x   : [B, H, T]
          cond: dict of conditioning tensors
          t   : [B] long; all entries are the current reverse step
        """
        pivot = max(cond.keys())
        c0, c1 = self.cmd_slice

        x_cpu   = x.detach().to("cpu").to(self.dtype, copy=True)        # [B,H,T]
        cmd_cpu = cond[pivot][:, c0:c1].detach().to("cpu", copy=True)   # [B,3]
        t_vec   = torch.full((x_cpu.shape[0],), int(t[0].item()), dtype=torch.int32)

        if self._f is not None:
            torch.save({"x": x_cpu, "cmd": cmd_cpu, "t": t_vec}, self._f)
            self._f.flush()
            return

        # legacy shard mode
        self._x_list.append(x_cpu)
        self._cmd_list.append(cmd_cpu)
        self._t_list.append(t_vec)
        self._rows += x_cpu.shape[0]
        if self._rows >= self.flush_every_rows:
            self.flush()

    def flush(self):
        if self._f is not None or not self._x_list:
            return
        x = torch.cat(self._x_list, dim=0)
        cmd = torch.cat(self._cmd_list, dim=0)
        tt = torch.cat(self._t_list, dim=0)
        stamp = int(time.time() * 1000)
        fname = os.path.join(self.out_dir, f"tele_{self.tag}_{self._shard_idx:06d}_{stamp}.pt")
        torch.save({"x": x, "cmd": cmd, "t": tt}, fname)
        self._x_list.clear(); self._cmd_list.clear(); self._t_list.clear()
        self._rows = 0; self._shard_idx += 1

    def close(self):
        if self._f is not None and not self._f.closed:
            self._f.flush(); self._f.close(); self._f = None