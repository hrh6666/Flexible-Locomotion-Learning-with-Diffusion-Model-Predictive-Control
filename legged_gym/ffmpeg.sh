ffmpeg -f image2 -start_number 0 -i logs/images/%4d.png -vframes 2500 -vsync 0 -c:v libx265 -r 50 output.mp4
ffmpeg -y -i output.mp4 -c:v libx264 -c:a aac -strict experimental -tune fastdecode -pix_fmt yuv420p -b:a 192k -ar 48000 out.mp4
