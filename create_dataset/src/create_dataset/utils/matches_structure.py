def matches_structure(data, structure):
    """ 응답이 기대하는 구조를 따르는지 확인하는 함수 """
    if isinstance(data, list) and isinstance(structure, list):
        if all(isinstance(item, dict) for item in data):
            return True
    elif isinstance(data, dict) and isinstance(structure, dict):
        return all(key in data and isinstance(data[key], type(structure[key])) for key in structure)
    return False