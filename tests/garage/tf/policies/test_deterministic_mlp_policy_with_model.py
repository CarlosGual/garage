import pickle
from unittest import mock

from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import DeterministicMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.models import SimpleMLPModel


class TestDeterministicMLPPolicyWithModel(TfGraphTestCase):
    @params(
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    )
    def test_get_action(self, obs_dim, action_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'deterministic_mlp_policy_with_model.MLPModel'),
                        new=SimpleMLPModel):
            policy = DeterministicMLPPolicyWithModel(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        action, _ = policy.get_action(obs)

        expected_action = np.full(action_dim, 0.5)

        assert env.action_space.contains(action)
        assert np.array_equal(action, expected_action)

        actions, _ = policy.get_actions([obs, obs, obs])
        for action in actions:
            assert env.action_space.contains(action)
            assert np.array_equal(action, expected_action)

    @params(
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    )
    def test_get_action_sym(self, obs_dim, action_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'deterministic_mlp_policy_with_model.MLPModel'),
                        new=SimpleMLPModel):
            policy = DeterministicMLPPolicyWithModel(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        obs_dim = env.spec.observation_space.flat_dim
        state_input = tf.placeholder(tf.float32, shape=(None, obs_dim))
        action_sym = policy.get_action_sym(state_input, name='action_sym')

        expected_action = np.full(action_dim, 0.5)

        action = self.sess.run(
            action_sym, feed_dict={state_input: [obs.flatten()]})
        action = policy.action_space.unflatten(action)

        assert np.array_equal(action, expected_action)
        assert env.action_space.contains(action)

    @params(
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    )
    def test_is_pickleable(self, obs_dim, action_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'deterministic_mlp_policy_with_model.MLPModel'),
                        new=SimpleMLPModel):
            policy = DeterministicMLPPolicyWithModel(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        with tf.variable_scope('DeterministicMLPPolicy/MLPModel', reuse=True):
            return_var = tf.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())
        output1 = self.sess.run(
            policy.model.outputs,
            feed_dict={policy.model.input: [obs.flatten()]})

        p = pickle.dumps(policy)
        with tf.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            output2 = sess.run(
                policy_pickled.model.outputs,
                feed_dict={policy_pickled.model.input: [obs.flatten()]})
            assert np.array_equal(output1, output2)
