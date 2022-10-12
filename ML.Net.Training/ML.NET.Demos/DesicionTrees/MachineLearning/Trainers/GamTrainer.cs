using DesicionTrees.MachineLearning.Common;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;

namespace DesicionTrees.MachineLearning.Trainers
{
    public class GamTrainer : TrainerBase<CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>
    {
        public GamTrainer() : base()
        {
            Name = "GAM";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .Gam();
        }
    }
}
