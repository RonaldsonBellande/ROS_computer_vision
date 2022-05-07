from header_imports import *


class classification_with_model(object):
    def __init__(self, saved_model):
        
        self.pointcloud = []
        self.pointcloud_data = []
        self.number_of_points = 2048
        self.saved_model = saved_model
        self.model = keras.models.load_model("models/" + self.saved_model)
        self.path  = "PointCloud_data/"
        self.true_path = self.path + "Testing/"
        self.number_images_to_plot = 16
        self.valid_images = [".off"]
        self.graph_path = "graph_charts/" + "prediction_with_model_saved/"
        self.model_category = ['toilet', 'monitor', 'dresser', 'sofa', 'table', 'night_stand', 'chair', 'bathtub', 'bed', 'desk']
        
        self.setup_structure()
        self.plot_prediction_with_model()


    def setup_structure(self):

        self.category_names =  os.listdir(self.true_path)
        folder = next(os.walk(self.true_path))[1]
        self.number_classes = len(folder)
        
        for i in range(self.number_classes):
            self.check_valid(self.category_names[i])
        
        for label in self.category_names:
            self.pointcloud_file = [self.true_path + label + '/' + i for i in os.listdir(self.true_path + '/' + label)]
            for point in self.pointcloud_file:
                self.pointcloud.append(trimesh.load(point).sample(self.number_of_points))
                vertice, face = self.vertices_and_faces(point)
                self.pointcloud_data.append((vertice, face))
        
        self.pointcloud = np.array(self.pointcloud)
        self.pointcloud_data = np.array(self.pointcloud_data)
        self.pointcloud =  self.pointcloud.reshape(self.pointcloud.shape[0], self.pointcloud.shape[1], self.pointcloud.shape[2], 1)
        self.X_test = self.pointcloud.astype("float32") / 255


    def check_valid(self, input_file):

        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue


    def plot_prediction_with_model(self):

        fig=plt.figure(dpi=500)
        predicted_classes = self.model.predict(self.X_test)
        axis=fig.add_subplot(111, projection='3d')
         
        for i in range(self.number_images_to_plot):
            plt.subplot(4,4,i+1)
            print("1: ", self.pointcloud_data[i])
            print("2: ", self.pointcloud_data[i][0])
            print("3: ", self.pointcloud_data[i][1])
            axis.plot_trisurf(self.pointcloud_data[i][0][:, 0], self.pointcloud_data[i][0][:,1], triangles=self.pointcloud_data[i][1], Z=self.pointcloud_data[i][0][:,2])
            plt.show()
            plt.axis('off')
            plt.title("Predicted - {}".format(self.model_category[np.argmax(predicted_classes[i], axis=0)]), fontsize=1)
            plt.tight_layout()
            plt.savefig(self.graph_path + "model_classification_detection_with_model_trained_prediction" + str(self.saved_model) + '.png')


    
    def vertices_and_faces(self, file_name):
        with open(file_name, 'r') as file:
            if 'OFF' != file.readline().strip():
                raise('Not a valid OFF header')
            
            n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
            vertices = [[float(s) for s in file.readline().strip().split(' ')] for i in range(n_verts)]
            faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i in range(n_faces)]

            return  np.array(vertices), np.zeros((len(faces)))


        
