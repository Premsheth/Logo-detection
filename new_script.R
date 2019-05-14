setwd("C:/Users/admin/Desktop/Data Analytics/logo_detection/New_data/")
options(stringsAsFactors = F)
df <- read.csv("flickr_logos_27_dataset_query_set_annotation.txt", sep="\t")
require(keras)

### Xception Transfer-learning example
img_width <- 75
img_height <- 75
batch_size <- 8

train_directory <- "flickrData/train"
test_directory <- "flickrData/test"

train_generator <- flow_images_from_directory(train_directory, generator = image_data_generator(),
                                              target_size = c(img_width, img_height), color_mode = "rgb",
                                              class_mode = "categorical", batch_size = batch_size, shuffle = TRUE,
                                              seed = 123)

validation_generator <- flow_images_from_directory(test_directory, generator = image_data_generator(),                                                   target_size = c(img_width, img_height), color_mode = "rgb", classes = NULL, class_mode = "categorical", batch_size = batch_size, shuffle = TRUE, seed = 123)


train_samples = 498
validation_samples = 177

########### generator to enhance the dataset, artificially alter the original data ################
datagen <- image_data_generator(
  rotation_range = 20,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  horizontal_flip = TRUE
)

train_augmented_generator <-  flow_images_from_directory(test_directory, generator = datagen,
                                                         target_size = c(img_width, img_height), color_mode = "rgb", classes = NULL, class_mode = "categorical", batch_size = batch_size, shuffle = TRUE,  seed = 123)

## define the pretrained model, here: Xception 
base_model <- application_xception(weights = 'imagenet', include_top = FALSE, input_shape = c(img_width, img_height, 3))



predictions <- base_model$output %>% 
  layer_global_average_pooling_2d(trainable = T) %>% 
  layer_dense(64, trainable = T) %>%
  layer_activation("relu", trainable = T) %>%
  layer_dropout(0.4, trainable = T) %>%
  layer_dense(27, trainable=T) %>%    ## important to adapt to fit the 27 classes in the dataset!
  layer_activation("softmax", trainable=T)


# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)

#################
for (layer in base_model$layers)
  layer$trainable <- FALSE

summary(model)
###################
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.003, decay = 1e-6),
  metrics = "accuracy"
)

hist <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(train_samples/batch_size), 
  epochs = 100, 
  validation_data = validation_generator,
  validation_steps = as.integer(validation_samples/batch_size),
  verbose=2
)

evaluate_generator(model,validation_generator, validation_samples)

###################### Train on augmented: artificially altered data #######
hist_aug <- model %>% fit_generator(
  train_augmented_generator,
  steps_per_epoch = as.integer(train_samples/batch_size), 
  epochs = 50, 
  validation_data = validation_generator,
  validation_steps = as.integer(validation_samples/batch_size),
  verbose=2
)
evaluate_generator(model,validation_generator, validation_samples)

img <- image_load("HP.jpg", target_size = c(75,75))
x <- image_to_array(img)
dim(x) <- c(1, dim(x)) 
prediction <- model %>% predict(x)

colnames(prediction) <- unique(df[,2])[1:27]
prediction[,which.max(prediction)]
