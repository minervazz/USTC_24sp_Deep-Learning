# src/tasks/concept_normalization.py

def normalize_concept(concept):
    normalization_dict = {
        'usb 3.0': 'usb 3.x',
        'usb 3.1 gen 1': 'usb 3.x',
        'usb 3.2 gen 1': 'usb 3.x',
        'usb 5g': 'usb 3.x'
    }
    return normalization_dict.get(concept.lower(), concept)
