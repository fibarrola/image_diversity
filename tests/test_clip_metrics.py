from image_diversity import ClipMetrics
import pytest

clip_metrics = ClipMetrics()


class TestClipMetrics:
    def test_tcd_negis(self):
        """TCE should not work if the number of eigenvalues is smaller than the number of images"""
        clip_metrics.n_eigs = 5
        with pytest.raises(AssertionError):
            clip_metrics.tce("tests/image_set_1")

    def test_tcd_output(self):
        """TCE should produce a positive value"""
        clip_metrics.n_eigs = 2
        assert clip_metrics.tce("tests/image_set_1") > 0, "TCE value should be positive"

    def test_tcd_subset(self):
        """TCE works with an image subset"""
        clip_metrics.n_eigs = 2
        assert (
            clip_metrics.tce("tests/image_set_1", ["A_10.png", "A_30.png", "A_50.png"])
            > 0
        ), "TCE value should be positive"

    def test_fcd_negis(self):
        """FCD should not work if the number of eigenvalues is smaller than the number of images"""
        clip_metrics.n_eigs = 5
        with pytest.raises(AssertionError):
            clip_metrics.fcd("tests/image_set_1", "tests/image_set_2")

    def test_fcd_output(self):
        """FCD should not work if the number of eigenvalues is smaller than the number of images"""
        clip_metrics.n_eigs = 2
        assert (
            clip_metrics.fcd("tests/image_set_1", "tests/image_set_2") > 0
        ), "FCD value should be positive"

    def test_fcd_diff_subsets(self):
        """Warning should be raised if FCD is comparing sets of different sizes"""
        clip_metrics.n_eigs = 1
        with pytest.warns(Warning):
            clip_metrics.fcd(
                "tests/image_set_1",
                "tests/image_set_2",
                ["A_10.png", "A_30.png", "A_50.png"],
                ["V_10.png", "V_30.png"],
            )

    def test_fcd_subsets(self):
        """FCD should produce a positive value"""
        clip_metrics.n_eigs = 2
        assert (
            clip_metrics.fcd(
                "tests/image_set_1",
                "tests/image_set_2",
                ["A_10.png", "A_30.png", "A_50.png"],
                ["V_10.png", "V_30.png", "V_50.png"],
            )
            > 0
        )
