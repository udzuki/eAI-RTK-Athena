"""Test for Athena."""

import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np

import pytest

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf
from repair.methods.athena import Athena


@pytest.fixture(scope="function")
def resource_root(tmp_path):
    return shutil.copytree(Path("tests/resources/fashion-mnist"), tmp_path / "outputs")


@pytest.fixture(scope="function")
def negative_root(resource_root):
    return resource_root / "negative"


@pytest.fixture(scope="function")
def positive_root(resource_root):
    return resource_root / "positive"


@pytest.fixture(scope="function")
def model(resource_root):
    return load_model_from_tf(resource_root / "model")


@pytest.fixture(scope="function")
def athena():
    common_settings = {"num_particles": 2, "num_iterations": 2}

    athena = Athena()
    athena.set_options(**common_settings)
    return athena


@pytest.fixture(scope="function")
def personal_best_scores(athena, model, weights, positive_root, negative_root):
    input_pos = athena.load_input_pos(positive_root)
    input_pos_sampled = athena._sample_positive_inputs(input_pos)
    input_neg = athena.load_input_neg(negative_root / "0")

    locations = athena._get_initial_particle_positions(weights, model, athena.num_particles)
    personal_best_scores = athena._initialize_personal_best_scores(
        locations, model, input_pos_sampled, input_neg
    )

    return personal_best_scores


@pytest.fixture(scope="function")
def weights(athena, setup_weight_label, negative_root):
    loaded_weights = athena.load_weights(negative_root / "0")
    _weights = []
    for w in loaded_weights:
        _weights += loaded_weights[w]["weights"]
    return _weights


@pytest.fixture(scope="function")
def best_particle(personal_best_scores):
    return np.argmax(np.array(personal_best_scores)[:, 0])


@pytest.fixture(scope="function")
def history(personal_best_scores, best_particle):
    return [personal_best_scores[best_particle]]


@pytest.fixture(scope="function")
def setup_no_weights(negative_root):
    labels = [
        [
            "6",
            {
                "repair_priority": 0,
                "prevent_degradation": 0,
            },
        ]
    ]

    with open(negative_root / "0" / "labels.json", "w") as f:
        json.dump(labels, f)


@pytest.fixture()
def setup_weight_label(negative_root):
    """Evacutate and restore the original file."""
    labels = [
        [
            "6",
            {
                "repair_priority": 1,
                "prevent_degradation": 0,
                "weights": [[15, 126, 0], [15, 405, 6], [15, 419, 0], [15, 201, 6], [15, 419, 6]],
            },
        ]
    ]

    with open(negative_root / "0" / "labels.json", "w") as f:
        json.dump(labels, f)


@pytest.fixture()
def setup_weights_csv(negative_root):
    weights = [
        [15, 126, 0, 0.17492177, -0.09925158],
        [15, 405, 6, 0.19923395, -0.11959197],
        [15, 419, 0, 0.17959525, 0.17334044],
        [15, 201, 6, 0.14921123, -0.15762208],
        [15, 419, 6, 0.19604559, -0.074185394],
    ]

    with open(negative_root / "0" / "6" / "weights.csv", "w") as f:
        w = csv.writer(f)
        w.writerows(weights)


@pytest.fixture(scope="function")
def setup_weights_optimized(negative_root):
    labels = [
        [
            "6",
            {
                "repair_priority": 1,
                "prevent_degradation": 0,
                "weights": [[15, 126, 0], [15, 405, 6], [15, 419, 0], [15, 201, 6], [15, 419, 6]],
                "repaired_values": [
                    "0.025723966",
                    "0.098222084",
                    "0.08759517",
                    "-0.14765175",
                    "0.010384405",
                ],
            },
        ]
    ]

    with open(negative_root / "0" / "labels.json", "w") as f:
        json.dump(labels, f)


class AthenaTest:
    """Test class for Athena."""

    def test_load_input_neg_one(self, athena, negative_root):
        """Load data of only one negative label.

        This test checks that the function loads a correct dataset.
        This test uses a directory whose labels.json has only one setting that
        "repair_priority: 1" which means "Reading only one data file".
        """
        imgs, labels = athena.load_input_neg(negative_root)
        # Check the type of loaded input
        assert type(imgs) == np.ndarray
        assert type(labels) == np.ndarray

        # Check whether the number of imgs and labels are the same
        assert len(imgs) == len(labels)

        # Check whther the number of imgs and labels are same as the target
        dataset = RepairDataset.load_repair_data(negative_root / "0")
        expected_imgs = dataset[0]
        expected_labels = dataset[1]

        assert len(imgs) == len(expected_imgs)
        assert len(labels) == len(expected_labels)

    def test_load_input_neg_multi(self, athena, negative_root):
        """Load data of only one negative label.

        This test checks that the function loads a correct dataset.
        This test uses a directory whose labels.json has two settings that
        "repair_priority: 1" which means "Reading two data file".

        """
        imgs, labels = athena.load_input_neg(negative_root / "2")

        # Check the type of loaded input
        assert type(imgs) == np.ndarray
        assert type(labels) == np.ndarray

        # Check whether the number of imgs and labels are the same
        assert len(imgs) == len(labels)

        dataset = RepairDataset.load_repair_data(negative_root / "2")
        expected_imgs = dataset[0]
        expected_labels = dataset[1]

        assert len(imgs) == len(expected_imgs)
        assert len(labels) == len(expected_labels)

    def test_load_input_neg_raise_error_when_empty(self, athena, setup_no_weights, negative_root):
        with pytest.raises(ValueError):
            athena.load_input_neg(negative_root / "0")

    def test_load_input_pos(self, athena, positive_root):
        """Load data of only one negative label.

        This test checks that the function loads a correct dataset.
        """
        imgs, labels = athena.load_input_pos(positive_root)

        # Check the type of loaded input
        assert type(imgs) == np.ndarray
        assert type(labels) == np.ndarray

        # Check whether the number of imgs and labels are the same
        assert len(imgs) == len(labels)

        # Check whether loading all the positive inputs
        dataset = RepairDataset.load_repair_data(positive_root)
        expected_imgs = dataset[0]
        expected_labels = dataset[1]
        assert len(imgs) == len(expected_imgs)
        assert len(labels) == len(expected_labels)

        # Check whether setting protected inputs with the designated rate
        target_data = positive_root / "0"

        dataset = RepairDataset.load_repair_data(target_data)
        expected_imgs = dataset[0]
        expected_labels = dataset[1]

        protected_imgs, protected_labels = athena.input_protected
        assert len(protected_imgs) == len(expected_imgs)
        assert len(protected_labels) == len(expected_labels)

    def test_load_weights(self, athena, setup_weight_label, resource_root):
        """Load weights."""
        weights = athena.load_weights(resource_root / "negative" / "0")

        # Check whether loading the correct weights
        assert len(weights) == 1
        assert weights["6"]["weights"] == [
            [15, 126, 0],
            [15, 405, 6],
            [15, 419, 0],
            [15, 201, 6],
            [15, 419, 6],
        ]

    def test_sample_positive_inputs_designated(self, athena, positive_root):
        """Sample positive inputs with the designated number."""
        kwargs = {"num_input_pos_sampled": 20}
        athena.set_options(**kwargs)
        input_pos = athena.load_input_pos(positive_root)
        sampled = athena._sample_positive_inputs(input_pos)

        # Check the number of sampled inputs
        assert len(sampled[0]) == 20
        assert len(sampled[1]) == 20

    def test_sample_positive_inputs_none(self, athena, positive_root):
        """Do not sample positive inputs."""
        kwargs = {"num_input_pos_sampled": None}
        athena.set_options(**kwargs)
        input_pos = athena.load_input_pos(positive_root)
        sampled = athena._sample_positive_inputs(input_pos)

        np.testing.assert_allclose(input_pos[0], sampled[0])
        np.testing.assert_allclose(input_pos[1], sampled[1])

    def test_fail_to_find_better_patch(self, athena, personal_best_scores, best_particle, history):
        """Tests for fail_to_find_better_patch with several settings.

        Tests for this method is buched because the initialize for this
        method is time consuming.
        """
        # Return false because the iteration num is 1
        assert not athena._fail_to_find_better_patch(1, history)
        # Return True because failured in finding the better score
        for _ in range(10):
            history.append(personal_best_scores[best_particle])
        personal_best_scores[best_particle][1] = 1
        assert athena._fail_to_find_better_patch(11, history)

        # Return False because succeeded in finding the better score
        new_score = copy.copy(personal_best_scores[best_particle])
        new_score[0] = 200
        history.append(new_score)
        assert not athena._fail_to_find_better_patch(12, history)

        # Return False because failured in finding the better patch
        personal_best_scores[best_particle][1] = -1
        history.remove(new_score)
        assert not athena._fail_to_find_better_patch(11, history)

    def test_criterion(self, athena, model, negative_root, setup_weight_label, positive_root):
        """Return correct type values."""
        weights = athena.load_weights(negative_root / "0")
        _weights = []
        for w in weights:
            _weights = _weights + weights[w]["weights"]

        input_pos = athena.load_input_pos(positive_root)
        input_pos_sampled = athena._sample_positive_inputs(input_pos)
        input_neg = athena.load_input_neg(negative_root / "0")

        locations = athena._get_initial_particle_positions(_weights, model, athena.num_particles)

        print(len(athena.input_protected[0]))

        fitness, n_patched, n_intact = athena._criterion(
            model, locations[0], input_pos_sampled, input_neg
        )

        np.testing.assert_allclose(fitness, 184.05, rtol=1)
        assert isinstance(n_patched, int)
        assert isinstance(n_intact, int)

    def test_localize(self, athena, model, negative_root):
        """Execute localize soundly."""
        target_data_dir = negative_root / "0"
        input_neg = athena.load_input_neg(target_data_dir)

        athena.localize(model, input_neg, target_data_dir)

        req = athena._load_requirements(target_data_dir)

        for label in req:
            assert "weights" in req[label]
            assert isinstance(req[label]["weights"], list)

    def test_skipping_localize(self, athena, model, negative_root):
        """Skip localize function with the already localied weights."""
        target_data_dir = negative_root / "0"
        input_neg = athena.load_input_neg(target_data_dir)

        athena.localize(model, input_neg, target_data_dir)

    def test_optimize(
        self,
        athena,
        model,
        setup_weight_label,
        setup_weights_csv,
        negative_root,
        positive_root,
        resource_root,
    ):
        """Execute optimize soundly."""
        target_data_dir = negative_root / "0"
        weights = athena._load_requirements(target_data_dir)
        input_pos = athena.load_input_pos(positive_root)

        athena.optimize(
            model,
            resource_root / "model",
            weights,
            target_data_dir,
            input_pos,
            target_data_dir,
        )

        req = athena._load_requirements(target_data_dir)

        for label in req:
            assert "repaired_values" in req[label]
            assert len(req[label]["repaired_values"]) == len(req[label]["weights"])

    def test_skip_optimize(
        self, athena, setup_weights_optimized, resource_root, negative_root, positive_root
    ):
        """Skip optimize function with already optimized weights."""
        target_data_dir = negative_root / "0"
        model = load_model_from_tf(resource_root / "model")
        weights = athena._load_requirements(target_data_dir)
        input_pos = athena.load_input_pos(positive_root)

        athena.optimize(
            model,
            resource_root / "model",
            weights,
            target_data_dir,
            input_pos,
            target_data_dir,
        )

    def test_evaluate(self, athena):
        """Do nothing because evaluate is not implemented."""
        athena.evaluate(None, None, None, None, None, None, None, None)
