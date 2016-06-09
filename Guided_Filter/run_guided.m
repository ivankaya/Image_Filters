%%Run MATLAB-integrated guided filter
image_path = './images/';

%input_image = 'bird.jpg';
%input_image = 'noise.jpg';
input_image = 'academy.jpg';
%input_image = 'einstein.jpg';
%input_image = 'mandrill.jpg';



input_image = imread(strcat(image_path, input_image));
image_smooth = imguidedfilter(input_image, input_image, 'NeighborhoodSize', [5 5], 'DegreeOfSmoothing', 0.01*diff(getrangefromclass(input_image)).^2);

imshowpair(input_image, image_smooth, 'montage');