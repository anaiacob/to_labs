function[M]=Matrix_video(vid,frame1,frame2)
Img = read(vid,frame1); 
Img = rgb2gray(Img); % 360 x 640
M   = Img(:);
  for i=frame1+1:frame2 
   frames = read(vid,i); % read picture in time i
   frames = rgb2gray(frames);
   M=[M frames(:)];
  end
end
