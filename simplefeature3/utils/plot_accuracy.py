

def plot_training(accuracies_file):
    '''
    :param self:
    :param accuracies_file: path to train accuracy file
    :return:

    file format:
    each line contains two numbers, (train_step_num batch_accuracy)
    '''
    reader = open(accuracies_file, "r")
    accuracies = []

    x=[]
    for line in reader:
        tmp = line.strip().split()
        step_num, acc = int(tmp[0]), float(tmp[1])
        accuracies.append(acc)
        x.append(step_num)
    reader.close()

    import matplotlib.pyplot as plt
    plt.plot(x, accuracies)
    plt.xlabel('Training step')
    plt.ylabel('Accuracy')
    plt.ylim([0.1, 1.0])
    plt.title('cnn,accuracy')
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_acc_path', type=str, default='')

    args = parser.parse_args()
    import os
    if not os.path.isfile(args.train_acc_path):
        print("please indicate train accuracy file path")
        exit(0)

    plot_training(args.train_acc_path)