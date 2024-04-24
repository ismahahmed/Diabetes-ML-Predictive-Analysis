import data_exploration
import models

def main():
    
    print('\nData Exploration:\n')
    data_exploration.main()

    print('\nModel Results:\n')
    model_table, accuracy_dict = models.main()
    print(model_table,'\n')

    for model, accuracy in accuracy_dict.items():
        print(f"{model} had accuracy of {accuracy}")


if __name__ == "__main__":
    main()