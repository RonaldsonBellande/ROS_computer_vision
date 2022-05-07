from header_imports import *


class model_building(models):
    def __init__(self, model_type):

        self.pointcloud = []
        self.label_name = []
        self.image_size = 224
        self.number_of_points = 2048
        self.valid_images = [".off"]
        self.model_type = model_type
        self.model_summary = "model_summary/"
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.labelencoder = LabelEncoder()

        self.setup_structure()
        self.splitting_data_normalize()

        if self.model_type == "model1":
            self.model = self.create_models_1()
        elif self.model_type == "model2":
            self.model = self.create_models_2()
        elif self.model_type == "model3":
            self.model = self.create_model_3()

        self.save_model_summary()
    

    def setup_structure(self):
        
        self.path  = "PointCloud_data/"
        self.true_path = self.path + "PointCloud/"
        self.category_names =  os.listdir(self.true_path)
        self.number_classes = len(next(os.walk(self.true_path))[1])

        for i in range(0, self.number_classes):
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


    def save_model_summary(self):

        with open(self.model_summary + self.model_type +"_summary_architecture_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()


    



