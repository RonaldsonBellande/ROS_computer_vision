from header_imports import *


class transfer_learning(models):
    def __init__(self, saved_model, model_type, random_noise_count):
        
        self.image_file = []
        self.label_name = []
        self.random_noise_count = int(random_noise_count)
        self.image_size = 240
        self.saved_model = saved_model
        self.number_of_nodes = 16

        self.valid_images = [".jpg",".png"]
        self.model_summary = "model_summary/"
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.model_type = model_type
        
        self.labelencoder = LabelEncoder()
        self.setup_structure() 
        self.splitting_data_normalize()
        
        if self.model_type == "model1":
            self.model = self.create_models_1()
        elif self.model_type == "model2":
            self.model = self.create_models_2()
        elif self.model_type == "model3":
            self.model = self.create_model_3()

        self.model.load_weights("models/" + self.saved_model)
        self.batch_size = [10, 20, 40, 60, 80, 100]
        self.epochs = [1, 5, 10, 50, 100, 200]
        self.number_images_to_plot = 16
        self.graph_path = "graph_charts/transfer_learning_with_model/"
        self.model_path = "models/transfer_learning/" 
        self.param_grid = dict(batch_size=self.batch_size, epochs=self.epochs)
        self.callback_1 = TensorBoard(log_dir="logs/{}-{}".format(self.model_type, int(time.time())))
        self.callback_2 = ModelCheckpoint(filepath=self.model_path, save_weights_only=True, verbose=1)
        self.callback_3 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor= 0.5, min_lr=0.00001)
        
        self.train_model()
        self.evaluate_model()
        self.plot_model()
        self.plot_prediction_with_model()


    def setup_structure(self):
        
        self.path  = "vehicle_image_data/"
        self.true_path = self.path
        self.category_names =  os.listdir(self.true_path)
        self.number_classes = len(next(os.walk(self.true_path))[1])
            
        for i in range(self.number_classes):
            self.check_valid(self.category_names[i])

        for i in range(self.number_classes):
            self.resize_image_and_label_image(self.category_names[i])

        self.label_name = self.labelencoder.fit_transform(self.label_name)
        self.image_file = np.array(self.image_file)
        self.label_name = np.array(self.label_name)
        self.label_name = self.label_name.reshape((len(self.image_file),1))


    def check_valid(self, input_file):
        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue
    

    def resize_image_and_label_image(self, input_file):
        for image in os.listdir(self.true_path + input_file):
            image_resized = cv2.imread(os.path.join(self.true_path + input_file,image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file.append(image_resized)
            self.label_name.append(input_file)
            self.adding_random_noise(image_resized, input_file)


    def adding_random_noise(self, image, input_file):
        
        # Gaussian noise 
        for i in range(self.random_noise_count):
            gaussian_noise = np.random.normal(0, (10 **0.5), image.shape)
            image = image + gaussian_noise
            self.image_file.append(image)
            self.label_name.append(input_file)


        # Salt and pepper noise 
        for i in range(self.random_noise_count):
            probability = 0.02
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    random_num = random.random()
                    if random_num < probability:
                        image[i][j] = 0
                    elif random_num > (1 - probability):
                        image[i][j] = 255
            self.image_file.append(image)
            self.label_name.append(input_file)


        # Poisson noise
        for i in range(self.random_noise_count):
            poisson_noise = np.sqrt(image) * np.random.normal(0, 1, image.shape)
            noisy_image = image + poisson_noise
            self.image_file.append(image)
            self.label_name.append(input_file)


        # Speckle noise
        for i in range(self.random_noise_count):
            speckle_noise = np.random.normal(0, (10 **0.5), image.shape)
            image = image + image * speckle_noise
            self.image_file.append(image)
            self.label_name.append(input_file)


        # Uniform noise
        for i in range(self.random_noise_count):
            uniform_noise = np.random.uniform(0,(10 **0.5), image.shape)
            image = image + uniform_noise
            self.image_file.append(image)
            self.label_name.append(input_file)



    def splitting_data_normalize(self):
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.image_file, self.label_name, test_size=0.10, random_state=42)
        self.input_shape = self.X_train.shape[1:]
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
        self.X_train = self.X_train.astype("float32") /255
        self.X_test = self.X_test.astype("float32") / 255


    def train_model(self):
       
        grid = GridSearchCV(estimator = self.model, param_grid = self.param_grid, n_jobs = 1, cv = 3, verbose = 10)

        self.computer_vision_model = self.model.fit(self.X_train, self.Y_train,
                batch_size=self.batch_size[0],
                validation_split=0.15,
                epochs=self.epochs[3],
                callbacks=[self.callback_1, self.callback_2, self.callback_3],
                shuffle=True)

        self.model.save(self.model_path + self.model_type + "_computer_vision_categories_"+ str(self.number_classes)+"_model.h5")
   

    def evaluate_model(self):
        evaluation = self.model.evaluate(self.X_test, self.Y_test, verbose=1)

        with open(self.graph_path + self.model_type + "_evaluate_computer_vision_category_" + str(self.number_classes) + ".txt", 'w') as write:
            write.writelines("Loss: " + str(evaluation[0]) + "\n")
            write.writelines("Accuracy: " + str(evaluation[1]))
        
        print("Loss:", evaluation[0])
        print("Accuracy: ", evaluation[1])


    def plot_model(self):

        plt.plot(self.computer_vision_model.history['accuracy'])
        plt.plot(self.computer_vision_model.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig(self.graph_path + self.model_type + '_accuracy_' + str(self.number_classes) + '.png', dpi =500)
        plt.clf()

        plt.plot(self.computer_vision_model.history['loss'])
        plt.plot(self.computer_vision_model.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig(self.graph_path + self.model_type + '_lost_' + str(self.number_classes) +'.png', dpi =500)
        plt.clf()


    def plot_prediction_with_model(self):

        plt.figure(dpi=500)
        predicted_classes = self.model.predict(self.X_test)
        
        for i in range(self.number_images_to_plot):
            plt.subplot(4,4,i+1)
            fig=plt.imshow(self.X_test[i,:,:,:])
            plt.axis('off')
            plt.title("Predicted - {}".format(self.category_names[np.argmax(predicted_classes[i], axis=0)]) + "\n Actual - {}".format(self.category_names[np.argmax(self.Y_test_vec[i,0])]),fontsize=1)
            plt.tight_layout()
            plt.savefig(self.graph_path + "model_detection_localization_with_model_trained_prediction_" + str(self.saved_model) + '.png')

        
