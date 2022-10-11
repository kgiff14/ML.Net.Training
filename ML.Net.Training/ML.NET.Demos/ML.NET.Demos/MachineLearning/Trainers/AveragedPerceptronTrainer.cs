using BinaryClassification.MachineLearning.Common;
using Microsoft.ML.Trainers;

namespace BinaryClassification.MachineLearning.Trainers
{
    public class AveragedPerceptronTrainer : TrainerBase<LinearBinaryModelParameters>
    {
        public AveragedPerceptronTrainer() : base()
        {
            Name = "Average Percption";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
