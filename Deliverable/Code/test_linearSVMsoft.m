function [ class_label, other_values ] = test_linearSVMsoft( data, model )
    class_label = sign([data,ones(size(data,1),1)]*model);
    other_values = -1;
end

