# src/tasks/elaboration.py

def elaborate_concept(concept):
    elaboration_dict = {
        'usb 3.x': 'USB 3.x refers to a series of specifications for USB interfaces.'
    }
    return elaboration_dict.get(concept.lower(), "No elaboration available.")
