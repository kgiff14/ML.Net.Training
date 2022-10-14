using RandomForestRegression.MachineLearning.Common;
using RandomForestRegression.MachineLearning.DataModels;
using RandomForestRegression.MachineLearning.Predictors;
using RandomForestRegression.MachineLearning.Trainers;

namespace RandomForestRegression
{
    public class Program
    {
        static void Main(string[] args)
        {
            var newSample = new HousingData
            {
                Age = 58.7f,
                CrimeRate = 0.02985f,
                EmployCenterDistance = 6.0622f,
                HighwayAccessabilityRadius = 3f,
                NOConcentration = 0.458f,
                NumOfRoomsPerDwelling = 6.43f,
                Proportion = 2.18f,
                PTRatio = 18.7f,
                RiverCoast = 0,
                TaxRate = 222f,
                Zoned = 0f
            };

            var trainers = new List<ITrainerBase>
            {
                new RandomForestTrainer(3,3),
                new RandomForestTrainer(3,8),
                new RandomForestTrainer(2,15),
                new RandomForestTrainer(5,2),
                new RandomForestTrainer(2,46)
            };

            trainers.ForEach(x => TrainEvalutePredict(x, newSample));
        }

        static void TrainEvalutePredict(ITrainerBase trainer, HousingData sample)
        {
            Console.WriteLine($"\n\n-------------------------------------------\n\t{trainer.Name}\n-------------------------------------------");

            string fileName = "boston_housing.csv";
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Data\", fileName);

            trainer.Fit(path.ToString());

            var modelMetrics = trainer.Evalute();

            Console.WriteLine($"\nLoss Function: {modelMetrics.LossFunction: 0.##}");
            Console.WriteLine($"\nMean Absolute Error: {modelMetrics.MeanAbsoluteError: #.##}");
            Console.WriteLine($"\nMean Squared Error: {modelMetrics.MeanSquaredError: #.##}");
            Console.WriteLine($"\nRSquared: {modelMetrics.RSquared: 0.##}");
            Console.WriteLine($"\nRoot Mean Squared Error: {modelMetrics.RootMeanSquaredError: 0.##}");

            trainer.Save();

            var predictor = new Predictor();
            var prediction = predictor.Predict(sample);
            Console.WriteLine($"\nPrediction: {prediction.MedianPrice:#.##}");
        }
    }
}