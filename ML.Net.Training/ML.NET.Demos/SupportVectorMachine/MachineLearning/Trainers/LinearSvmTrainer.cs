using Microsoft.ML.Trainers;
using SupportVectorMachineClassification.MachineLearning.Common;

namespace SupportVectorMachineClassificiation.MachineLearning.Trainers
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
