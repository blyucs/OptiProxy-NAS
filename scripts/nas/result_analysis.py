import os
import numpy as np
import matplotlib.pyplot as plt
import sys
















































































































































































def main(results_dir):
    l = os.scandir(results_dir)
    l = sorted(l, key=lambda e: e.name)
    acc_list = []
    test_acc_list = []

    results_file = os.path.join(results_dir, 'results_summary.txt')
    with open(results_file, 'w') as f:
        for entry in l:
            if entry.is_dir():  
                log_file = os.path.join(entry, "log.log")
                if os.path.isfile(log_file):  
                    with open(log_file, 'r') as ll:
                        lines = ll.readlines()
                        last_line = lines[-1]
                        f.write(f"{entry.name}: {last_line}")
                        if float(last_line.split()[-1]) > 0:
                            acc_list.append(float(last_line.split()[-1]))
                        if float(last_line.split()[-3]) > 0:
                            test_acc_list.append(float(last_line.split()[-3]))

        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)

        f.write(f"\nMean Accuracy: {mean_acc}\n")
        f.write(f"Standard Deviation of Accuracy: {std_acc}\n")
        print(f"\nMean Accuracy: {mean_acc}\n")
        print(f"Standard Deviation of Accuracy: {std_acc}\n")

        mean_test_acc = np.mean(test_acc_list)
        std_test_acc = np.std(test_acc_list)

        f.write(f"\nMean test Accuracy: {mean_test_acc}\n")
        f.write(f"Standard Deviation of test Accuracy: {std_test_acc}\n")
        print(f"\nMean test Accuracy: {mean_test_acc}\n")
        print(f"Standard Deviation of test Accuracy: {std_test_acc}\n")

    plt.scatter(np.arange(len(acc_list)), acc_list)
    plt.xlabel('Experiment Index')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over different seeds')
    plot_file = os.path.join(results_dir, 'accuracy_scatter_plot.png')
    plt.savefig(plot_file)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python results_analysis.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    
    main(results_dir)










