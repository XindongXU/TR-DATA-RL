# TR_DATA_ReinforcementLearning
A data science and machine learning project proposed by Mines Paris Sophia, control of an underwater robotic arm by reinforcement learning approach.
This paper proposes a deep reinforcement learning-based controller for a flexible underwater robotic arm. Due to the complex and harsh underwater environment, traditional robot control methods face technical challenges such as high pressure, reduced visibility, and limited communication. To address these challenges, we apply deep reinforcement learning to train the robotic arm to make optimal decisions based on its environment without constant human intervention. Our experiments demonstrate that our approach can achieve an accuracy rate of $82.8\%$ in reaching operator-specified locations. However, our research still faces some limitations such as imprecise position detection and mechanical instability. Future work will focus on addressing these limitations and extending our approach to more complex underwater operations.

### /project_RL/DQ_Learning.py/
* In this script, we create python class `DQNet`, a tensorflow keras model class, which gets input of current state information, given target position and the action chosen by the algorithm, and tries to predict the Q state action value, which presents the quality of such an action under this circumstance.
* Also, a class named `environment`, an environment which updates at every time step after the algorithm of robotic arm taking action to the current situation.
* A python function `save_memory`, who is going to save all the enivornment parameters into an experience storage, which will be used during replay learning, and the trainning of network and ensure that the size of repaly memory is smaller than the given size.
* A python function `GetMinibatch`, who randomly sample some components from replay memory storage, with a length of given value.

### /project_RL/DQ_Learning_train.py/
* In this script, we import necessary python class and functions `environment`, `DQNet`, `LearningRateReducerCb`, `GetMinibatch`,  from script `DQ_Learning.py` and create a double Q network RL training environment.

### /project_RL/reset_position.py/

### /project_RL/webcam_connect.py/

### /project_RL/joystick.py/
