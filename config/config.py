def load_config():
    """
    Returns a dict with:
      - paths to model files
      - thresholds (confidence, detection)
      - question definitions file path
    """
    return {
        'asl_model': 'models/hand_kp_best_A_Z.h5',
        'asl_map':   'models/hand_kp_class_indices_A_Z.json',
        'digit_model': 'models/kp_only_best_1_9.h5',
        'digit_map':   'models/kp_class_indices_1_9.json',
        'confidence_threshold': 0.6,
        'questions_file': 'questions.py',
    }