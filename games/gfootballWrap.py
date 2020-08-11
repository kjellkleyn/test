import datetime
import os

import gym
import numpy
import torch

import gfootball.env as football_env


class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        ### Game
        self.observation_shape = (4, 96, 72)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(18)]  # Fixed list of all possible actions. You should only edit the length
        self.players = [i for i in range(1)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_device = "cpu"  # "cpu" / "cuda"
        self.selfplay_num_gpus = 0  # Number of GPUs per actor to use for the selfplay, it can be fractional, don't fortget to take the training worker, the test worker and the other selfplay workers into account. (ex: if you have 1 GPU and num_workers=1 -> selfplay_num_gpus=1/3 because 1/3 for the training, 1/3 for test worker selfplay and 1/3 for this selfplay worker)
        self.max_moves = 700  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size

        # Residual Network
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.blocks = 16  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 128  # Number of channels in reward head
        self.reduced_channels_value = 128  # Number of channels in value head
        self.reduced_channels_policy = 128  # Number of channels in policy head
        self.resnet_fc_reward_layers = [128, 128]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [128, 128]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [128, 128]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [1024, 512, 128, 64]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [1024, 512, 128, 64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64,64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64,64]  # Define the hidden layers in the value network
        self.fc_policy_layers = [64,64]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_weights = True  # Save the weights in results_path as model.weights
        self.training_steps = 25000 # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 20  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.training_device = "cpu"  # Train on GPU if available. "cpu" / "cuda"
        self.training_num_gpus = 0  # Number of GPUs to use for the training, it can be fractional, don't fortget to take the test worker and the selfplay workers into account

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 100



        ### Replay Buffer
        self.window_size = 100  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 32  # Number of game moves to keep for every batch element
        self.td_steps = 10  # Number of steps in the future to take into account for calculating the target value
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Prioritized Replay (See paper appendix Training)
        self.PER = True  # Select in priority the elements in the replay buffer which are unexpected for the network
        self.use_max_priority = False  # If False, use the n-step TD error as initial priority. Better for large replay buffer
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_beta = 1.0



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game():
    """
    Game wrapper.
    """
    def __init__(self, render = False, seed=None):
        self.env = football_env.create_environment(
            env_name='academy_empty_goal_close', 
            stacked=False,
            representation='extracted',
            logdir='/tmp/rllib_test',
            write_goal_dumps=False, write_full_episode_dumps=False, 
            render=False,
            dump_frequency=0,
            number_of_left_players_agent_controls=1)
        self.action_space = [i for i in range(18)]  # Fixed list of all possible actions. You should only edit the length
        print("env obs shape: ", self.env.observation_space)
        if seed is not None:
            self.env.seed(seed)
    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return 0


    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _ = self.env.step(action)
        
        return numpy.reshape(numpy.array(observation), (4, 96, 72)), reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return self.action_space

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        obs = numpy.array(self.env.reset())
        return numpy.reshape(obs, (4, 96, 72))


    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        #self.env.render()
        pass
    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "action_idle", 
            1: "action_left",
            2: "action_top_left",
            3: "action_top",
            4: "action_top_right",
            5: "action_right",
            6: "action_bottom_right",
            7: "action_bottom",
            8: "action_bottom_left",
            9: "action_long_pass",
            10: "action_high_pass",
            11: "action_short_pass",
            12: "action_shot",
            13: "action_sprint",
            14: "action_release_direction",
            15: "action_release_sprint",
            16: "action_sliding",
            17: "action_dribble",
            18: "action_release_dribble",
        }
        return "{}. {}".format(action_number, actions[action_number])