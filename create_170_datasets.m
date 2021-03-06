function create_170_datasets
num_features  = 10;
num_instances = 500;
%rng('default')
for k = 1 : 100
 x = randn(num_instances,1);
 y = randn(num_instances,1);
 z = randn(num_instances,1);
 labels = [];
 
 for i = 1 :size(x,1)    
     if sqrt( (x(i)-0)^2 + (y(i)-0)^2 ) < 2 &&  sqrt( (x(i)-0)^2 + (z(i)-0)^2 ) <0.9 && x(i) > 0         
         labels(i) = 1;
     else                 
         labels(i) = 2;
     end          
 end  
 
 
data = [labels' x,y,z];
   
TRAIN_class_labels = data(:,1);
TRAIN = data(:,2:end);
TEST_class_labels = data(:,1);
TEST = data(:,2:end);

correct = 0; % Initialize the number we got correct

for i = 1 : length(TEST_class_labels) % Loop over every instance in the test set
   classify_this_object = TEST(i,:);
   this_objects_actual_class = TEST_class_labels(i);
   predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels, classify_this_object,i);
   if predicted_class == this_objects_actual_class
       correct = correct + 1;
   end;   
end;


data = randn(num_instances,num_features);
index = randperm(num_features);
disp(['On this dataset ',int2str(k) ', the accuracy can be ',num2str(1-(length(TEST_class_labels)-correct )/length(TEST_class_labels),2), ' when using only features ', int2str( index(1:3))])

 data(:,index(1)) = x;
 data(:,index(2)) = y;
 data(:,index(3)) = z; 
 data = [labels' data];
 
 %%%%
 
 
TRAIN_class_labels = data(:,1);
TRAIN = data(:,2:end);
TEST_class_labels = data(:,1);
TEST = data(:,2:end);

correct = 0; % Initialize the number we got correct

for i = 1 : length(TEST_class_labels) % Loop over every instance in the test set
   classify_this_object = TEST(i,:);
   this_objects_actual_class = TEST_class_labels(i);
   predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels, classify_this_object,i);
   if predicted_class == this_objects_actual_class
       correct = correct + 1;
   end;   
end;
 
disp(['But using all the features the accuracy will be just ',num2str(1-(length(TEST_class_labels)-correct )/length(TEST_class_labels),2)])
%%%
 
 
 
 
 
 
 
 
 str = ['save Ver_2_CS170_Fall_2021_Small_data__',int2str(k) ,'.txt data -ASCII' ];
 eval(str)
 
 

end











function predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels,unknown_object, dont_compare)
best_so_far = inf;
 for i = 1 : length(TRAIN_class_labels)     
     if i ~= dont_compare         
     compare_to_this_object = TRAIN(i,:);
     distance = sqrt(sum((compare_to_this_object - unknown_object).^2)); % Euclidean distance

        if distance < best_so_far
          predicted_class = TRAIN_class_labels(i);
          best_so_far = distance;
        end
        
     end 
end;
 
 
 
 
 
 
 
  
 
 
