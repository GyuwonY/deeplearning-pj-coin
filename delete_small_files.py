import os

def delete_small_files():
    daycandle_dir = 'daycandle'
    if not os.path.exists(daycandle_dir):
        return

    row_sum = 0
    for filename in os.listdir(daycandle_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(daycandle_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                row_count = sum(1 for row in f)
            
            row_sum += row_count
    
    print(row_sum)
    

if __name__ == "__main__":
    delete_small_files()