# src/tasks/relational_inference.py

def infer_relationship(entity1, entity2):
    relationships = {
        ('product', 'category'): 'belongs to',
        ('category', 'attribute'): 'has attribute'
    }
    return relationships.get((entity1, entity2), 'no relationship found')
