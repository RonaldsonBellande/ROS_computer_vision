from header_imports import *


class transfer_learning(models):
    def __init__(self, saved_model, model_type):
        
        self.pointcloud = []
        self.label_name = []
        self.saved_model = saved_model
        self.model_type = model_type
        self.batch_size = [10, 20, 40, 60, 80, 100]
        self.epochs = [1, 5, 15, 50, 100, 200]
        self.number_of_points = 2048
        self.model_path = "models/transfer_learning/" 
        self.number_images_to_plot = 16
        self.valid_images = [".off"]
        self.labelencoder = LabelEncoder()
        self.graph_path = "graph_charts/transfer_learning_with_model/"
        
        self.setup_structure()
        self.splitting_data_normalize()
        
        if self.model_type == "model1":
            self.model = self.create_models_1()
        elif self.model_type == "model2":
            self.model = self.create_models_2()
        elif self.model_type == "model3":
            self.model = self.create_model_3()

        self.model.load_weights("models/" + self.saved_model)
        self.param_grid = dict(batch_size = self.batch_size, epochs = self.epochs)
        self.callback_1 = TensorBoard(log_dir="logs/{}-{}".format(self.model_type, int(time.time())))
        self.callback_2 = ModelCheckpoint(filepath=self.model_path, save_weights_only=True, verbose=1)
        self.callback_3 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor= 0.5, min_lr=0.00001)
        
        self.train_model()
        self.evaluate_model()
        self.plot_model()
        self.plot_prediction_with_model()


    def setup_structure(self):
        
        self.path  = "PointCloud_data/"
        self.true_path = self.path + "PointCloud/"
        self.category_names =  os.listdir(self.true_path)
        self.number_classes = len(next(os.walk(self.true_path))[1])
        
        for i in range(self.number_classes):
            self.check_valid(self.category_names[i])
        
        for label in self.category_names:
            self.pointcloud_file = [self.true_path + label + '/' + i for i in os.listdir(self.true_path + '/' + label)]
            for point in self.pointcloud_file:
                self.pointcloud.append(trimesh.load(point).sample(self.number_of_points))
                self.label_name.append(label)
        
        self.label_name = self.labelencoder.fit_transform(self.label_name)
        self.pointcloud = np.array(self.pointcloud)
        self.pointcloud =  self.pointcloud.reshape(self.pointcloud.shape[0], self.pointcloud.shape[1], self.pointcloud.shape[2], 1)
        self.label_name = np.array(self.label_name)
        self.label_name = tf.keras.utils.to_categorical(self.label_name , num_classes=self.number_classes)


    def check_valid(self, input_file):

        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue

    def splitting_data_normalize(self):
        
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.pointcloud, self.label_name, test_size = 0.10, random_state = 42)
        self.input_shape = self.X_train.shape[1:]
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
        self.X_train = self.X_train.astype("float32") / 255
        self.X_test = self.X_test.astype("float32") / 255

    

    def train_model(self):
       
        grid = GridSearchCV(estimator = self.model, param_grid = self.param_grid, n_jobs = 1, cv = 3, verbose = 10)

        self.computer_vision_model = self.model.fit(self.X_train, self.Y_train_vec,
                batch_size=self.batch_size[4],
                validation_split=0.10,
                epochs=self.epochs[3],
                callbacks=[self.callback_1, self.callback_2, self.callback_3],
                shuffle=True)

        self.model.save(self.model_path + self.model_type + "_computer_vision_categories_"+ str(self.number_classes)+"_model.h5")
   

    def evaluate_model(self):
        evaluation = self.model.evaluate(self.X_test, self.Y_test_vec, verbose=1)

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
            plt.axis('off')
            plt.title("Predicted - {}".format(self.category_names[np.argmax(predicted_classes[i], axis=0)]) + "\n Actual - {}".format(self.category_names[np.argmax(self.Y_test_vec[i,0])]),fontsize=1)
            plt.tight_layout()
            plt.savefig(self.graph_path + "model_classification_detection_with_model_trained_prediction" + '.png')

        
