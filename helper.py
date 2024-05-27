import json


def save(obj: list, path: str) -> bool:
    try:
        with open(path, 'w', encoding='utf8') as f:
            json.dump(obj, f, ensure_ascii=False)
        f.close()
        return True
    except Exception as e:
        print('<<ERROR>>:', e)
        exit(0)


def load(path: str) -> dict | None:
    try:
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        f.close()
        return data
    except Exception as e:
        print('<<ERROR>>:', e)
        exit(0)


def cuda_info_getter():
    import torch
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))


if __name__ == '__main__':
    cuda_info_getter()
