using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Weight = System.Double;
using Neuron = System.Collections.Generic.List<System.Double>;
using Layer = System.Collections.Generic.List<System.Collections.Generic.List<System.Double>>;
using Weights = System.Collections.Generic.List<System.Collections.Generic.List<System.Collections.Generic.List<System.Double>>>;

using Matrix = System.Collections.Generic.List<System.Collections.Generic.List<System.Double>>;
using Vector = System.Collections.Generic.List<System.Double>;

public static class Extensions
{
    /*public static double[][] ToExtArray(this List<List<Double>> list)
    {
        return list.Select(a => a.ToArray()).ToArray();
    }*/

    public static List<Double> ToList(this double[] array)
    {
        List<Double> ret = new List<Double>();

        for(int i = 0; i < array.Length; i++)
        {
            ret.Add(array[i]);
        }

        return ret;
    }

    public static Vector getColumn(this Layer m, int col)
    {
        Vector ret = new Vector();

        for (int row = 0; row < m.Count; row++)
        {
            ret.Add(m[row][col]);
        }

        return ret;
    }

    public static Vector getRow(this Layer m, int row)
    {
        return m[row];
    }

    public static void print(this Layer m)
    {
        for (int row = 0; row < m.Count; row++)
        {
            Console.Write("|");
            for (int col = 0; col < m[0].Count; col++)
            {
                Console.Write(" {0}", m[row][col]);
            }
            Console.WriteLine(" |");
        }
        Console.WriteLine("");
    }

    public static void print(this Weights w)
    {
        foreach (var val in w) val.print();
    }

    public static List<List<Double>> ToExtList(this double[][] array)
    {
        List<List<Double>> list = new List<List<Double>>();
        for (int i = 0; i < array.Length; i++)
        {
            List<Double> l = array[i].ToList();
            list.Add(l);
        }

        return list;
    }
}

public class Util
{
    static System.Random random = new System.Random();

    public static Layer createZeroMatrix(int nRows, int nCols)
    {
        Layer ret = new Layer();

        for (int row = 0; row < nRows; row++)
        {
            ret.Add(new Vector());
            for (int col = 0; col < nCols; col++)
            {
                ret[row].Add(0.0);
            }
        }

        return ret;
    }

    public static double meanSquaredError(double[][] target, double[][] y)
    {
        int n = target.Length;

        double sum = 0.0;
        for (int i = 0; i < n; i++)
        {
            sum += Math.Pow((target[i][0] - y[i][0]), 2);
        }

        return 1.0 / n * sum;
    }

    public static double meanSquaredError(Matrix target, Matrix y)
    {
        int n = target.Count;

        double sum = 0.0;
        for (int i = 0; i < n; i++)
        {
            sum += Math.Pow((target[i][0] - y[i][0]), 2);
        }

        return 1.0 / n * sum;
    }

    public static double dot(Vector a, Vector b)
    {
        double total = 0.0;
        for (int i = 0; i < a.Count; i++)
        {
            total += a[i] * b[i];
        }

        return total;
    }

    public static Layer matMultiply(Layer m1, Layer m2)
    {
        int nRows = m1.Count;
        int nCols = m2[0].Count;

        Layer ret = new Layer();

        for (int row = 0; row < nRows; row++)
        {
            ret.Add(new Vector());
            for (int col = 0; col < nCols; col++)
            {
                ret[row].Add(dot(m1.getRow(row), m2.getColumn(col)));
            }
        }

        return ret;
    }

    public static double sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    public static Layer sigmoid(Layer m)
    {
        Layer newM = new Layer();

        for (int row = 0; row < m.Count; row++)
        {
            newM.Add(new Vector());
            for (int col = 0; col < m[0].Count; col++)
            {
                newM[row].Add(sigmoid(m[row][col]));
            }
        }

        return newM;
    }

    public static List<double> getRandomDouble(int num, double min = -1.0, double max = 1.0)
    {
        List<double> values = new List<double>();
        for (int i = 0; i < num; i++)
        {
            values.Add(randomFloatRange(min, max));
        }

        return values;
    }

    public static double randomFloatRange(double min, double max)
    {
        return random.NextDouble() * (max - min) + min;
    }

    public static Weights cloneWeights(Weights weights)
    {
        Weights newWeights = new Weights();
        for (int l = 0; l < weights.Count; l++)
        {
            newWeights.Add(new Layer());
            for (int n = 0; n < weights[l].Count; n++)
            {
                newWeights[l].Add(new Neuron());
                for (int w = 0; w < weights[l][n].Count; w++)
                {
                    newWeights[l][n].Add(weights[l][n][w]);
                }
            }
        }

        return newWeights;
    }

    public static double gaussianNoise(double mu = 0, double sigma = 1)
    {
        //return randomFloatRange(-1.0, 1.0);
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        double randNormal = mu + sigma * randStdNormal;

        return randNormal;
    }
}

class NN
{
    public int[] topology;
    public Weights weights;
    public Matrix input;
    public Matrix y;
    public Matrix output;

    public NN(Matrix x, Matrix y_)
    {
        topology = new int[] { 2, 2, 1 };
        weights = new Weights();
        input = x;
        y = y_;
        output = Util.createZeroMatrix(y.Count, y[0].Count);

        randomiseWeights();
    }

    public NN(double[][] x, double[][] y_)
    {
        topology = new int[] { 2, 2, 1 };
        weights = new Weights();
        input = x.ToExtList();
        y = y_.ToExtList();
        output = Util.createZeroMatrix(y.Count, y[0].Count);

        randomiseWeights();
    }

    public Layer feedForward()
    {
        var lastOut = input;

        for (int i = 0; i < topology.Length - 1; i++)
        {
            Layer val = Util.matMultiply(lastOut, weights[i]);
            lastOut = Util.sigmoid(val);
        }

        output = lastOut;

        return output;
    }

    public Layer feedForwardW(Weights weightsIn)
    {
        var lastOut = input;

        for (int i = 0; i < topology.Length - 1; i++)
        {
            lastOut = Util.sigmoid(Util.matMultiply(lastOut, weightsIn[i]));
        }

        return lastOut;
    }

    public void randomiseWeights()
    {
        weights = new Weights();

        for (int l = 0; l < topology.Length - 1; l++)
        {
            weights.Add(new Layer());
            for (int n = 0; n < topology[l]; n++)
            {
                weights[l].Add(new Neuron());
                for (int w = 0; w < topology[l + 1]; w++)
                {
                    weights[l][n].Add(Util.randomFloatRange(-1.0, 1.0));
                }
            }
        }
    }

    public Vector predict(Vector input)
    {
        Layer lastOut = new Layer() { input };

        for (int i = 0; i < topology.Length - 1; i++)
        {
            Layer val = Util.matMultiply(lastOut, weights[i]);
            lastOut = Util.sigmoid(val);
        }

        return lastOut[0];
    }

    public void ehc(int nEpochs)
    {
        Console.WriteLine("Training");

        double errorGoal = 0.0;

        var weightsChamp = Util.cloneWeights(weights);
        var outputChamp = feedForwardW(weightsChamp);
        double errorChamp = Util.meanSquaredError(y, outputChamp);

        int counter = 0;

        while ((errorGoal <= errorChamp) && (counter <= nEpochs))
        {
            var weightsMutant = Util.cloneWeights(weightsChamp);
            double stepSize = 0.01 * Util.gaussianNoise();

            for (int l = 0; l < weights.Count; l++)
            {
                for (int i = 0; i < weights[l].Count; i++)
                {
                    for (int j = 0; j < weights[l][i].Count; j++)
                    {
                        var deltaWeight = stepSize * Util.gaussianNoise();
                        weightsMutant[l][i][j] += deltaWeight;
                    }
                }
            }

            var outputMutant = feedForwardW(weightsMutant);
            var errorMutant = Util.meanSquaredError(y, outputMutant);
            if (errorMutant < errorChamp)
            {
                weightsChamp = Util.cloneWeights(weightsMutant);
                errorChamp = errorMutant;
                //Console.WriteLine("Best error: {0}", errorChamp);
            }

            counter++;
        }

        weights = Util.cloneWeights(weightsChamp);
    }
}

public class EHC : MonoBehaviour
{
    NN nn;
    // Start is called before the first frame update
    void Start()
    {
        nn = new NeuralNetwork();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        
    }
}
