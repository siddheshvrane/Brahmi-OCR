import numpy as np

def mock_ensemble_logic(model_preds, model_accuracies, num_classes):
    num_crops = len(model_preds[next(iter(model_preds))])
    ensemble_probs = np.zeros((num_crops, num_classes))
    weight_sums = np.zeros(num_crops)

    for m_key, probs in model_preds.items():
        global_acc = model_accuracies.get(m_key, 0.90)
        for i in range(num_crops):
            local_conf = np.max(probs[i])
            dynamic_weight = global_acc * local_conf
            ensemble_probs[i] += dynamic_weight * probs[i]
            weight_sums[i] += dynamic_weight

    final_probabilities = np.zeros((num_crops, num_classes))
    for i in range(num_crops):
        if weight_sums[i] > 0:
            final_probabilities[i] = ensemble_probs[i] / weight_sums[i]
    
    return final_probabilities

# Test Case 1: Model A is generally more accurate, but Model B is more confident on this specific char.
# Model A: Acc 0.99, Conf 0.4 (uncertain)
# Model B: Acc 0.91, Conf 0.9 (certain)
accs = {'ModA': 0.99, 'ModB': 0.91}
preds = {
    'ModA': np.array([[0.4, 0.3, 0.3]]), # Class 0 top but low conf
    'ModB': np.array([[0.05, 0.9, 0.05]]) # Class 1 top and high conf
}

final = mock_ensemble_logic(preds, accs, 3)
print(f"Test Case 1 Probs: {final}")
top_class = np.argmax(final[0])
print(f"Top Class 1: {top_class} (Expected: 1 since ModB is much more certain)")

# Test Case 2: Both confident, Model A should win due to higher accuracy.
preds2 = {
    'ModA': np.array([[0.95, 0.025, 0.025]]), # Class 0
    'ModB': np.array([[0.025, 0.95, 0.025]])  # Class 1
}
final2 = mock_ensemble_logic(preds2, accs, 3)
print(f"Test Case 2 Probs: {final2}")
top_class2 = np.argmax(final2[0])
print(f"Top Class 2: {top_class2} (Expected: 0 because ModA is generally more accurate)")
