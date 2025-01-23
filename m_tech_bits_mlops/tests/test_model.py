from m_tech_bits_mlops.src.model import run_ml_model


def test_run_ml_model():
    output = run_ml_model()
    assert output == "Model Output"
