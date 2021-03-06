from header_imports import *


class continuous_learning(deep_q_learning, classification_enviroment, plot_graphs):
    def __init__(self, saved_model, model_type, episode, noise=0.0, reward_noise=0.0, state_world_size=400, algorithm_name="deep_q_learning", transfer_learning="true"):
        
        self.algorithm_details_path = "graph_charts/"
        self.algorithm_details = self.algorithm_details_path + "algorithm_details/"
        self.model_detail = self.algorithm_details_path + "model_details/"
        self.graph_path = self.algorithm_details_path + "continuous_learning_with_models/"
        
        self.dense_size = 10
        self.exploration_decay = 0.95
        self.pointcloud = []
        self.label_name = []
        self.saved_model = saved_model
        self.model_type = model_type
        self.batch_size = [10, 20, 40, 60, 80, 100]
        self.epochs = [1, 5, 15, 50, 100, 200]
        self.number_of_points = 2048
        self.model_path = "models/continuous_learning/" 
        self.number_images_to_plot = 16
        self.valid_images = [".off"]
        self.labelencoder = LabelEncoder()
        
        self.setup_structure()
        self.splitting_data_normalize()

        self.image_per_episode = int(math.sqrt(len(self.pointcloud)))
        self.train_initial_model = "false"
        self.algorithm_name = algorithm_name
        self.transfer_learning = transfer_learning
        self.episode = episode
        self.step_limit = 10
        self.epsilon = 1
        self.delay_epsilon = 0.995
        self.min_epsilon = 0.001
        self.episode_rewards = []
        self.step_per_episode = []

        deep_q_learning.__init__(self, saved_model=self.saved_model, model_type=self.model_type, dense_size=self.dense_size, batch_size=self.batch_size[3], exploration_decay=self.exploration_decay, algorithm_name=self.algorithm_name, transfer_learning=self.transfer_learning)
        classification_enviroment.__init__(self, number_classes=self.number_classes, data_set=(self.pointcloud, self.label_name), image_per_episode=self.image_per_episode)

        if self.algorithm_name == "deep_q_learning":
            self.deep_q_learning()
        elif self.algorithm_name == "double_deep_q_learning":
            self.double_deep_q_learning()
        elif self.algorithm_name == "dueling_deep_q_learning":
            self.dueling_deep_q_learning()

        plot_graphs.__init__(self)



    def setup_structure(self):
        
        self.path  = "PointCloud_data/"
        self.true_path = self.path + "PointCloud_Additional/"
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


    def deep_q_learning(self):
    
        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state, done = self.reset(), False
            episode_reward = 0

            for i in tqdm(range(1, self.image_per_episode), desc="image_per_episode"):
                action, reward, next_state, done = self.step(self.model(state[None])[0])
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, done))
                state = next_state
                self.memory_delay()
                step += 1
            
            self.train_initial_model = "true"
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        self.plot_episode_time_step(self.episode_rewards,type_graph="cumulative_reward")
        self.plot_episode_time_step(self.step_per_episode, type_graph="step_number")
        # self.plot_model()
        self.plot_prediction_with_model()


    def double_deep_q_learning(self):

        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state, done = self.reset(), False
            episode_reward = 0

            for i in tqdm(range(1, self.image_per_episode), desc="image_per_episode"):
                action, reward, next_state, done = self.step(self.model(state[None])[0])
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, done))
                state = next_state
                self.target_model_update(done)
                self.memory_delay()
                step += 1
                
            self.train_initial_model = "true"
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        self.plot_episode_time_step(self.episode_rewards,type_graph="cumulative_reward")
        self.plot_episode_time_step(self.step_per_episode, type_graph = "step_number")
        # self.plot_model()
        self.plot_prediction_with_model()


    def dueling_deep_q_learning(self):

        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state, done = self.reset(), False
            episode_reward = 0
            
            for i in tqdm(range(1, self.image_per_episode), desc="image_per_episode"):
                action, reward, next_state, done = self.step(self.model(state[None])[0])
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, done))
                state = next_state
                self.target_model_update(done)
                self.memory_delay()
                step += 1

            self.train_initial_model = "true"
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        self.plot_episode_time_step(self.episode_rewards, type_graph="cumulative_reward")
        self.plot_episode_time_step(self.step_per_episode, type_graph = "step_number")
        # self.plot_model()
        self.plot_prediction_with_model()


