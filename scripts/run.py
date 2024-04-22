import data_exploration
import models

def main():
    
    print('\nData Exploration:\n')
    data_exploration.main()

    print('\nModel Results:\n')
    model_results = models.main()
    print(model_results)


if __name__ == "__main__":
    main()