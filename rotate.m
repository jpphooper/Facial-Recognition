for i = 73
    str = sprintf('IMG_02%d.JPG',i);
    I = imrotate(imread(str),270);
    imwrite(I,str);
end