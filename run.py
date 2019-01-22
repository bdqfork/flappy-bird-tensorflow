import cv2
import numpy as np

import game.wrapped_flappy_bird as game
import RLBrain
from RLBrain import DeepQNetwork

ISTRAIN = False


# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(
        observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80, 1))


def playFlappyBird():
    # Step 1: init BrainDQN
    brain = DeepQNetwork()
    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1, 0])  # do nothing
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(
        observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    # Step 3.2: run the game
    time_step = 0
    score = 0
    while True:
        action, q_max = brain.getAction()
        nextObservation, reward, terminal = flappyBird.frame_step(action)
        nextObservation = preprocess(nextObservation)
        if reward == 1:
            score += 1
        elif reward == -1:
            print("game over score: %d" % score)
            if not ISTRAIN:
                exit()
            score = 0
        loss = brain.setPerception(
            time_step, action, reward, nextObservation, terminal)
        print_info(action, brain, loss, q_max, reward, time_step)
        time_step += 1


def print_info(action, brain, loss, q_max, reward, time_step):
    if time_step <= RLBrain.OBSERVE:
        state = "observe"
    elif RLBrain.OBSERVE < time_step <= RLBrain.OBSERVE + RLBrain.EXPLORE:
        state = "explore"
    else:
        state = "train"
    if q_max is None:
        print("TIMESTEP", time_step, "/ STATE", state,
              "/ EPSILON", brain.epsilon, "/ ACTION", np.argmax(action),
              "/ REWARD", reward)
    else:
        print("TIMESTEP", time_step, "/ STATE", state,
              "/ EPSILON", brain.epsilon, "/ ACTION", np.argmax(
                  action), "/ REWARD", reward,
              "/ Q_MAX %e" % np.max(q_max), "/ LOSS", loss)


def main():
    playFlappyBird()


if __name__ == '__main__':
    main()
