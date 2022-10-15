namespace ModelBuilder
{
    public class Program
    {
        public static void Main(string[] args)
        {
            //Load sample data
            var sampleData = new ModelBuilder.ModelInput()
            {
                Island = @"Torgersen",
                Culmen_length_mm = 39.1F,
                Culmen_depth_mm = 18.7F,
                Flipper_length_mm = 181F,
                Body_mass_g = 3750F,
                Sex = @"MALE",
            };

            //Load model and predict output
            var result = ModelBuilder.Predict(sampleData);

            Console.WriteLine(result.Prediction);
        }
    }
}

