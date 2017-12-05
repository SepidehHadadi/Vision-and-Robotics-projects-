function Vector = classifier(Vector)


if size(Vector,1) == 1
    Size = size(Vector, 2);
else
    Size = size(Vector, 1);
end


        
for i = 1 : Size
       if sqrt((Vector(i) - 0)^2) <  sqrt((Vector(i) - 1)^2)
       Vector(i) = 0;
       else 
       Vector(i) = 1;
       end
end
        
    
    
    
end