import random
import os
import random
# out_dir = 'code'

valid_list = ['problem056', 'problem122', 'problem037', 
        'problem033', 'problem102', 'problem289', 'problem257', 
        'problem047', 'problem178', 'problem111']

def create_dacon_test_pair(code_folder, out_file):
    problem_folders = os.listdir(code_folder)
    problem_folders = list(set(problem_folders) - set(valid_list))
    # problem_folders = random.sample(problem_folders, 10)
    # print(problem_folders)
    all_codes = []
    for problem_folder in problem_folders:
        scripts = os.listdir(os.path.join(code_folder,problem_folder))
        problem_number = scripts[0].split('_')[0]        
        code_paths = [(os.path.join(problem_folder, script), problem_number) for script in scripts]
        all_codes.extend(code_paths)
    
    f = open(out_file, 'w')
    count = 0
    for i, code_i in enumerate(all_codes):
        for j, code_j in enumerate(all_codes):
            if i > j:
                continue
            label = int(code_i[1] == code_j[1])
            if label == 1:
                pos_pairs = code_i[0] + ' ' + code_j[0] + ' ' + str(label) + '\n'
                f.write(pos_pairs)
                count += 1
                # print('{} pairs'.format(count), pos_pairs[:-1])

                while True:
                    code_k = random.choice(all_codes)
                    label = int(code_i[1] == code_k[1])
                    if label == 0:
                        break
                neg_pairs = code_i[0] + ' ' + code_k[0] + ' ' + str(label) + '\n'
                f.write(neg_pairs)
                count += 1
                # print('{} pairs'.format(count), neg_pairs[:-1])    

    f.close()


if __name__ == "__main__":
    create_dacon_test_pair(code_folder='code', out_file='train_pair.txt')