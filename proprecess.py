
from config import Config
import json

if __name__ == '__main__':
    config = Config()
    with open(config.processed_data + 'round1_train_0907.json', "r", encoding='utf-8') as f:
        data = json.load(f)
    train_data = data[:-10]
    val_data = data[-10:]
    with open(config.processed_data + 'train.json', "w", encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(config.processed_data + 'val.json', "w", encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False)
