using Microsoft.ML;
using Microsoft.ML.Trainers;
using RecommendationSystem.MachineLearning.Common;

namespace RecommendationSystem.MachineLearning.Trainers
{
    public class MatrixFactorizationTrainer : TrainerBase
    {
        public MatrixFactorizationTrainer(int approximationRank, int nunberOfIterations, double learningRate = 0.1) : base()
        {
            Name = $"Matrix Factorization-{approximationRank}-{nunberOfIterations}";
            Model = MlContext
                        .Recommendation()
                        .Trainers
                        .MatrixFactorization(labelColumnName: "Label", matrixColumnIndexColumnName: "UserIdEncoded", matrixRowIndexColumnName: "MovieIdEncoded", approximationRank: approximationRank, learningRate: learningRate, numberOfIterations: nunberOfIterations);
        }
    }
}
