from dataclasses import dataclass
from helper import load


@dataclass
class CocoInfo:
    coco_info_path: str = 'coco_info/'

    def get_coco_labels(self) -> dict:
        data = load(self.coco_info_path + 'coco_labels.json')
        int_keys_data = {int(key): value for key, value in data.items()}
        return int_keys_data

    def get_coco_colors_pattern(self, *, default: bool = True) -> dict:
        file_name = 'colors_default_pattern.json' if default else 'colors_secondary_pattern.json'
        return load(self.coco_info_path + file_name)


def main():
    coco = CocoInfo()
    print(coco)
    print(coco.get_coco_labels())
    print(coco.get_coco_colors_pattern())
    print(coco.get_coco_colors_pattern(default=False))


if __name__ == '__main__':
    main()
