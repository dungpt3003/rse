package com.vu;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.matrix.Matrix;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.Random;
import java.util.Collections;
import java.util.Arrays;


public class SemiBinaryRSE extends Classifier {
    // training data : labelled (first) + unlabeled
    protected Instances m_data;

    protected Instances m_labeledData;

    // number of labelled instances
    protected int m_numLabeled;

    //the number of intances in the data
    protected int m_numInst;

    protected int m_numAttr;
    protected int m_numClass;

    protected double m_lambda; //coef for regularizer

    //the number of base classifiers
    protected int m_numBaseClassifier = 100;
    protected Classifier[] m_Classifiers;
    protected boolean m_useRandomSubspace = false;
    protected double[] m_alpha;
    protected Random m_r;

    protected boolean m_Debug = false;

    // the classifier should be binary
    protected Classifier m_baseClassifier;

    public void setBaseClassifier(Classifier cls) {
        m_baseClassifier = cls;
    }

    public void setNumBaseClassifier(int n) {
        m_numBaseClassifier = n;
    }

    public void setNumLabeled(int l) {
        m_numLabeled = l;
    }

    public void setLambda(double l) {
        m_lambda = l;
    }

    public void setR(Random r) {
        m_r = r;
    }

    public void setUseRandomSubspace(boolean ur) {
        m_useRandomSubspace = ur;
    }

    public void setWeights(double[] w) {
        for (int i = 0; i < m_numBaseClassifier; i++) {
            m_alpha[i] = w[i];
        }
    }

    public int getNumBaseClassifiers() {
        return m_numBaseClassifier;
    }

    public int getNumTrainingData() {
        return m_numInst;
    }

    public int getNumLabeledData() {
        return m_numLabeled;
    }

    public double getLambda() {
        return m_lambda;
    }

    public Instances getResampledData() {
        Instances d = m_data.resample(m_r);
        return d;
    }

    public Instances getResampledLabeled() {
        Instances d = m_labeledData.resample(m_r);
        return d;
    }

    public Instances getLabeledData() {
        return m_labeledData;
    }


    public SemiBinaryRSE() {
    }

    @Override
    public String getRevision() {
        return null;
    }

    public void buildClassifier(Instances da) throws Exception {
        m_data = makeBalanceData(da);

        m_data.setClassIndex(m_data.numAttributes() - 1);
        m_numInst = m_data.numInstances();

        m_numAttr = m_data.numAttributes();
        m_numClass = m_data.numClasses();

        m_Classifiers = getRandomClassifiers(m_labeledData, m_numBaseClassifier,
                m_r);

        m_alpha = new double[m_numBaseClassifier];
    }

    protected Instances makeBalanceData(Instances da) {
        int numP = 0;
        int numN = 0;
        for (int i = 0; i < da.numInstances(); i++) {
            if (da.instance(i).classValue() == 1) {
                numP++;
            } else {
                numN++;
            }
        }
        float frac = (m_numLabeled + 1.0f) / da.numInstances();
        int LdP = Math.round(frac * numP);
        int LdN = m_numLabeled - LdP;

        Instances LD = new Instances(da, m_numLabeled);
        Instances nLD = new Instances(da, da.numInstances() - m_numLabeled);

        for (int i = 0; i < da.numInstances(); i++) {
            if (da.instance(i).classValue() == 1) {
                if (LdP > 0) {
                    LD.add(da.instance(i));
                    LdP--;
                } else {
                    nLD.add(da.instance(i));
                }
            } else {
                if (LdN > 0) {
                    LD.add(da.instance(i));
                    LdN--;
                } else {
                    nLD.add(da.instance(i));
                }

            }
        }

        Instances d = new Instances(da, da.numInstances());
        for (int i = 0; i < LD.numInstances(); i++) {
            d.add(LD.instance(i));
        }

        for (int i = 0; i < nLD.numInstances(); i++) {
            d.add(nLD.instance(i));
        }

        m_labeledData = LD;

        return d;
    }

    protected Instances getLabeledInstances() {
        Instances d = new Instances(m_data, m_numLabeled);
        for (int i = 0; i < m_numLabeled; i++) {
            d.add(m_data.instance(i)); ;
        }
        return d;
    }

    /**
     * get the class distributions for ins
     * @param ins Instance
     * @return double[]
     * @throws Exception
     */
    public double[] distributionForInstance(Instance ins) throws Exception {
        double[] p = new double[2];

        double v = 0;

        for (int cls = 0; cls < m_numBaseClassifier; cls++) {
            if (m_alpha[cls] > 0) {
                double cval = m_Classifiers[cls].classifyInstance(ins);
                if (cval == 1.0) {
                    v += m_alpha[cls];
                } else {
                    v += -1 * m_alpha[cls];
                }
            }
        }
        if (v > 0) {
            p[1] = 1.0;
        } else {
            p[0] = 1.0;
        }
        return p;
    }

    /**
     * get the true labels of the labeled training examples
     * @return double[] the labels
     * @throws Exception
     */
    public double[] getTrueLabels() throws Exception {
        // get the labels
        double[] m_labels = new double[m_numInst];
        for (int i = 0; i < m_numInst; i++) {
            m_labels[i] = m_data.instance(i).classValue() == 0 ? -1.0 : 1.0;
        }
        return m_labels;
    }

    /**
     * get the true labels of the labeled training examples
     * @return double[] the labels
     * @throws Exception
     */
    public double[] getLabeledTrueLabels() throws Exception {
        // get the labels
        double[] m_labels = new double[m_numLabeled];
        for (int i = 0; i < m_numLabeled; i++) {
            m_labels[i] = m_labeledData.instance(i).classValue() == 0 ? -1.0 :
                    1.0;
        }
        return m_labels;
    }


    /**
     * get the matrix W for Laplacian
     * @return double[][]
     * @throws Exception
     */
    public double[][] getKernelLinkMatrix() throws Exception {
        // get the link matrix
        NominalToBinary n2b = new NominalToBinary();
        n2b.setInputFormat(m_data);
        Instances d = n2b.useFilter(m_data, n2b);
        d.setClassIndex(d.numAttributes() - 1);

//        Normalize nl = new Normalize();
//        nl.setInputFormat(d);
//        d = nl.useFilter(d, nl);
//        d.setClassIndex(d.numAttributes() - 1);

        Matrix LMat = new Matrix(m_numInst, m_numInst);
        RBFKernel krl = new RBFKernel();
//        double g = 0.5 / (d.numAttributes() - 1.0);
//        krl.setGamma(g);
        krl.buildKernel(d);

        for (int i = 0; i < m_numInst; i++) {
            for (int j = 0; j <= i; j++) {
                double vt = krl.eval(i, j, d.instance(i));
                LMat.set(i, j, vt);
                LMat.set(j, i, vt);
            }
        }
        return LMat.getArray();
    }

    /**
     * get the matrix W for Laplacian
     * @return double[][]
     * @throws Exception
     */
    public double[][] getLabeledKernelLinkMatrix() throws Exception {
        // get the link matrix
        NominalToBinary n2b = new NominalToBinary();
        n2b.setInputFormat(m_labeledData);
        Instances d = n2b.useFilter(m_labeledData, n2b);
        d.setClassIndex(d.numAttributes() - 1);

//        Normalize nl = new Normalize();
//        nl.setInputFormat(d);
//        d = nl.useFilter(d, nl);
//        d.setClassIndex(d.numAttributes() - 1);

        Matrix LMat = new Matrix(m_numLabeled, m_numLabeled);
        RBFKernel krl = new RBFKernel();
//        double g = 0.5 / (d.numAttributes() - 1.0);
//        krl.setGamma(g);
        krl.buildKernel(d);

        for (int i = 0; i < m_numLabeled; i++) {
            for (int j = 0; j <= i; j++) {
                double vt = krl.eval(i, j, d.instance(i));
                LMat.set(i, j, vt);
                LMat.set(j, i, vt);
            }
        }
        return LMat.getArray();
    }

    public double[][] getPredictions() throws Exception {
        return getPredictions(m_data);
    }

    public double[][] getLabeledPredictions() throws Exception {
        return getPredictions(m_labeledData);
    }


    public double[][] getPredictions(Instances data) throws Exception {
        // get the prediction matrix P
        int numInst = data.numInstances();
        Matrix Prd = new Matrix(m_numBaseClassifier, numInst);
        for (int i = 0; i < numInst; i++) {
            for (int b = 0; b < m_numBaseClassifier; b++) {
                double pv = m_Classifiers[b].classifyInstance(data.instance(i)) ==
                        1.0 ? 1.0 : -1.0;
                Prd.set(b, i, pv);
            }
        }
        return Prd.getArray();
    }


    /**
     * Generate random base classifers
     * @param data Instances all the labeled data
     * @param number int the number of base classifers to get
     * @param rnd int the random varaible
     * @return Classifier[] the classifiers
     */
    protected Classifier[] getRandomClassifiers(Instances data, int number,
                                                Random rnd) throws Exception {

        if (m_Debug) {
            System.out.println("Generating Random Base Classifiers!");
        }
        Classifier[] clfs = new Classifier[number];

        // the attributes indces
        Integer[] indices = new Integer[data.numAttributes() - 1];
        int classIndex = data.classIndex();
        int offset = 0;
        for (int i = 0; i < indices.length + 1; i++) {
            if (i != classIndex) {
                indices[offset++] = Integer.valueOf(i + 1);
            }
        }

        // bootstrap + random_subspace
        for (int i = 0; i < number; i++) {
            //inbag data
            Instances d = data.resample(rnd);

            if (m_useRandomSubspace) {
                // randomspace
                FilteredClassifier fc = new FilteredClassifier();
                fc.setClassifier(Classifier.makeCopy(m_baseClassifier));
                Remove rm = new Remove();
                rm.setOptions(new String[] {"-V", "-R",
                        randomSubSpace(indices,
                                Math.round(indices.length / 2 + 1),
                                classIndex + 1, rnd)});
                fc.setFilter(rm);
                fc.buildClassifier(d);
                clfs[i] = fc;
            } else {
                Classifier cc = m_baseClassifier.makeCopy(m_baseClassifier);
                cc.buildClassifier(d);
                clfs[i] = cc;
            }
        }
        return clfs;
    }

    /**
     * generates an index string describing a random subspace, suitable for
     * the Remove filter.
     *
     * @param indices		the attribute indices
     * @param subSpaceSize	the size of the subspace
     * @param classIndex		the class index
     * @param random		the random number generator
     * @return			the generated string describing the subspace
     */
    protected String randomSubSpace(Integer[] indices, int subSpaceSize,
                                    int classIndex, Random random) {
        Collections.shuffle(Arrays.asList(indices), random);
        StringBuffer sb = new StringBuffer("");
        for (int i = 0; i < subSpaceSize; i++) {
            sb.append(indices[i] + ",");
        }
        sb.append(classIndex);

        if (getDebug()) {
            System.out.println("subSPACE = " + sb);
        }

        return sb.toString();
    }
}
