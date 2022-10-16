using Microsoft.ML.Trainers;
using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class LinearSvmTrainer : TrainerBase<LinearBinaryModelParameters>
    {
        public LinearSvmTrainer() : base()
        {
            Name = "Linear Svm";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .LinearSvm();
        }
    }
}
