
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.Timer;

import de.jannlab.Net;
import de.jannlab.error.Error;
import de.jannlab.core.CellType;
import de.jannlab.data.Sample;
import de.jannlab.data.SampleSet;
import de.jannlab.generator.GenerateNetworks;
import de.jannlab.generator.LSTMGenerator;
import de.jannlab.generator.MLPGenerator;
import de.jannlab.io.Serializer;
import de.jannlab.learning.BasicNetLearningListener;
import de.jannlab.learning.NetLearning;
import de.jannlab.learning.Sampling;
import de.jannlab.math.StatisticsHelper;
import de.jannlab.misc.TimeCounter;
import de.jannlab.optimization.BasicOptimizationListener;
import de.jannlab.optimization.DifferentiableObjective;
import de.jannlab.optimization.Objective;
import de.jannlab.optimization.OptimizerListener;
import de.jannlab.optimization.optimizer.Adam;
import de.jannlab.optimization.optimizer.GradientDescent;


public class BouncingBall1D {
	private boolean random;
	private boolean train;
	private boolean trainmlp;

	private String[] in;
	private String[] out;
	private String hid;
	private Event[] eventarray;
	private int inlayer;
	private int outlayer;
	
	// trainingsparameter
	
	private  int    epochs;
	private  int  length;
	private  int trainsize;
	private  double learningrate;
    
    
	private  double beta1; // mt=b1mt-1+(1-b1)g(t).
	private  double beta2; // vt=b2vt-1+(1-b2)g2t.
	private  double epsilon; // avoids division by 0, smoothing
	private  boolean biascorrection; 

    // Anzeige parameter
	private  int ystretch;
	private int currepoch =0;
	private double[] ballin;
	private double[] ballinprev;
	private boolean feedback;
	private boolean online;
	private String weightString;
    private static TimeCounter TC = new TimeCounter();
    private static Random rnd = new Random(0L);
	
	public BouncingBall1D(boolean random, String[] in, String[] out, String hid, boolean train, boolean trainmlp, int epochs, int length, int trainsize, double learningrate, double beta1, double beta2, double epsilon, boolean biascorrection, int ystretch, boolean feedback, boolean online) throws ClassNotFoundException, IOException{
	  	this.random = random;
		this.in= in; //
	  	this.out= out; //
	  	this.hid= hid; //
	    this.train = train;
	    this.trainmlp  =trainmlp;
        this.epochs = epochs;
        this.length = length;
        this.trainsize = trainsize;
        this.learningrate = learningrate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.biascorrection = biascorrection;
        this.ystretch = ystretch;
        this.feedback = feedback;
        this.online = online;
        this.eventarray = new Event[]{new Event("Linker Bounce",0,-1,false,Color.gray), new Event("Rechter Bounce",0,1,true,Color.black)};
        inlayer=0;
        for(int i=0; i<in.length; i++){
        	if(in[i]=="zero"){
        		inlayer +=1;
        	}
        	if(in[i]=="pos"){
        		inlayer +=1;
        	}
        	if(in[i]=="noicepos"){
        		inlayer +=1;
        	}
        	if(in[i]=="speed"){
        		inlayer +=1;
        	}
        }
        outlayer = 0;
        for(int i=0; i<out.length; i++){
        	if(out[i]=="pos"){
        		outlayer +=1;
        	}
        	if(out[i]=="posnext"){
        		outlayer +=1;
        	}
        	if(out[i]=="speed"){
        		outlayer +=1;
        	}
        	if(out[i]=="event"){
        		outlayer +=eventarray.length;
        	}
        }
        ballin = new double[inlayer];
        ballinprev = new double[inlayer];
        run();
	}//end constructor
    

    
   
    
    public Sample generateSample(final int length) {
    	
    	
    	double start = 0.0;
    	double speed = 0.1;
    	if(random){
    		start = Math.random() * 1.8 -0.9;
    		speed = Math.random() * 0.4-0.2;
    	}
    	double inputnoice = 0.01;
    	int evnum = eventarray.length;
    	
    	//generieren der daten
    	double[] pos = new double[length];
    	double[] noicepos = new double[length];
        double[] speeds =  new double[length];
        double[] events = new double[length*evnum];
        
	 	int sincelastevent = 0;
	 	for(int t=0; t<length;t++){
	 		if(t==0) pos[0]= start;
	 		else pos[t]= pos[t-1]+speed;
	 		noicepos[t] = pos[t]+(Math.random()*2-1)*inputnoice;
	 		speeds[t]=speed;
	 		int eventnum = 0;
	 		for(Event ev : eventarray){
    			if(ev.uperbound){
    				if(pos[t]<ev.value){
    					eventnum++;
    					continue;
    				}
    			}else{
    				if(pos[t]>ev.value){
    					eventnum++;
    					continue;
    				}
    			}
    			//event ist ausgelöst
    			if(t-sincelastevent>1)
    			speed *=-1;
    			
    			for(int i=sincelastevent; i<=t; i++){
    				for(int j=0;j<evnum;j++){
    				if(j==eventnum)
    					events[evnum*i+j]=1;
    				else
    					events[evnum*i+j]=0;
    				}
    			}
    			sincelastevent = t;
	 	 } // end for each event
	 	} // for each t
        
        // befüllen der daten in die buffer
    	double[] data = new double[length*inlayer];
        double[] target =  new double[length*outlayer];
        //
        for(int t=0; t<length;t++){
            for(int i=0; i<in.length; i++){
            	if(in[i]=="zero"){
            		data[t*inlayer+i]=0;
            	}
            	if(in[i]=="pos"){
            		data[t*inlayer+i]=pos[t];
            	}
            	if(in[i]=="noicepos"){
            		data[t*inlayer+i]=noicepos[t];
            	}
            	if(in[i]=="speed"){
            		data[t*inlayer+i]=speeds[t];
            	}
            }
           
            for(int i=0; i<out.length; i++){
            	if(out[i]=="pos"){
            		target[t*outlayer+i]=pos[t];
            	}
            	if(out[i]=="posnext"){
            		if(t==length-1)
            		target[t*outlayer+i]=pos[t]+speeds[t];
            		else
            		target[t*outlayer+i]=pos[t+1];
            	}
            	if(out[i]=="speed"){
            		target[t*outlayer+i]=speeds[t];
            	}
            	if(out[i]=="event"){
            		for(int j=0;j<evnum;j++){
            	target[t*outlayer+i+j]=events[t*evnum+j];
            	}//end for each event
            }// end if event should go to out
            } // end for each outputtask
        } // end t
        
        
        return new Sample(data, target, inlayer, length, outlayer, length);
    }
    
    
    
    public  SampleSet generate(final int n, final int length) {
        SampleSet set = new SampleSet();
        //
        for (int i = 0; i < n; i++) {
            set.add(generateSample(length));
        }
        //
        return set;
    }
  

    
 
    public void run() throws IOException, ClassNotFoundException {
        //

        weightString = "1DIn";
        
        for(int i=0; i<in.length; i++){
        	weightString +=in[i];
        }
        weightString +=hid+"out";
        for(int i=0; i<out.length; i++){
        	weightString +=out[i];
        }
        weightString +="random"+random;
        weightString +="online"+online;
        String netString = "LSTM-"+inlayer+hid+"linear"+outlayer+"b";
        final Net net = GenerateNetworks.generateNet(netString);
        
        if(online)net.rebuffer(1);
        else net.rebuffer(length);
        

        final SampleSet trainset = generate(trainsize, length);
        final SampleSet testset  = generate(1, length);

        if(!train){
    		final double[] weights = Serializer.read(weightString+".weights");
            net.writeWeights(weights, 0);
    	}else{
          


          //
          // setup network.
          //
          net.rebuffer(trainset.maxSequenceLength());
          
          DifferentiableObjective obj = new DifferentiableObjective() {
              public double computeGradient(double[] args, int offset,
  					double[] grad, int gradoffset) { 
                  //
                  // the arguments for the fitness function are used
                  // as weights within the recurrent network.
                  //
            	  if(currepoch>0){ //skip first step for using already trained net
                      net.writeWeights(args, offset);
                  	}else{
                  		net.readWeights(args, offset);
                  	}
                  //
                  // then the error is computed based on the output
                  // of the network carrying the previously updated
                  // weights.
                  //
            	  
            	  if(online){
            		  if(currepoch ==0){
            			  trainset.shuffle(rnd);
                      generateOnlineSample(net,trainset.get(0));
            		  }

                  
                    
                    double error = produceNetoutputOnline(net);
                    
                  	
                  	
                  	net.computeGradient();
                    net.readGradWeights(grad, gradoffset);
                      currepoch++;
                	  
                      return error;
                	
            	  }else{
            	  
                  trainset.shuffle(rnd);
              	//washoutphase in gradienten mit rein?
              	final double[] input = trainset.get(0).input.data;
                final double[] target = trainset.get(0).target.data;
                final double[] netoutput = new double[target.length];
                
                double error = produceNetoutput(net, input,target,netoutput);
                
              	
              	
              	net.computeGradient();
                net.readGradWeights(grad, gradoffset);
                  currepoch++;
            	  
                  return error;
            	  } // end not alternative
            }
              //
             
           

			public int arity() {
                  //
                  // the number of weights of the network gives
                  // the arity of the fitness fuction.
                  //
                  return net.getWeightsNum();
              }
			public double compute(double[] arg0, int arg1) {
				return 0;
			}
			
          };
          File f = new File(weightString+".weights");
  		if(f.exists() && !f.isDirectory()) { 
  			final double[] weights = Serializer.read(weightString+".weights");
            net.writeWeights(weights, 0);
  		}else{
  			currepoch=1;
  		}
          //
          // setup trainer.
          //
     
    		
          final Adam optimizer = new Adam();
          optimizer.setLearningRate(learningrate);;
          optimizer.setBeta1(beta1);
          optimizer.setBeta2(beta2);
          optimizer.setEpsilon(epsilon);
          optimizer.setBiasCorrection(biascorrection);
          optimizer.setRnd(rnd);
          optimizer.setParameters(net.getWeightsNum());
          optimizer.updateObjective(obj);
         
          //
          // setup learning.
          optimizer.initialize();
      
          TC.reset();
          optimizer.addListener(new BasicOptimizationListener());
          optimizer.iterate(epochs, 0);
          //
          System.out.println(
              "training time: " + 
              TC.valueMilliDouble() +
              " ms."
          );
          double[] solution = new double[net.getWeightsNum()];
          optimizer.readBestSolution(solution, 0);
          net.writeWeights(solution, 0);
 
          Serializer.write(solution, weightString+".weights");
        
    	} // end if train
        //
        // evaluate learning success.
        //
        
        final BufferedImage img = new BufferedImage(
                800, length*ystretch, BufferedImage.TYPE_INT_ARGB
            );
            final Graphics2D imggfx = (Graphics2D)img.getGraphics();
            imggfx.setRenderingHint(
                RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON
            );
            final Stroke stroke = new BasicStroke(1.5f);
            imggfx.setStroke(stroke);
            imggfx.setBackground(Color.WHITE);
            //
            final JPanel canvas = new JPanel(){
                private static final long serialVersionUID = -5927396264951085674L;
                //
                @Override
                protected void paintComponent(Graphics gfx) {
                    super.paintComponent(gfx);
                    gfx.drawImage(img, 0, 0, null);
                }
            };
            //
            final Dimension canvasdim = new Dimension(
                img.getWidth(), img.getHeight()
            );
            //
            canvas.setPreferredSize(canvasdim);
            canvas.setSize(canvasdim);
            //
            imggfx.clearRect(0, 0, img.getWidth(), img.getHeight());
             final int[] timestep = {0};
            //  
            
            final JFrame frame = new JFrame("1D Bouncing Ball!. ");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            
            //
            frame.add(canvas);
            frame.pack();
            frame.setVisible(true);
            
            // view 2 
            final BufferedImage img2 = new BufferedImage(
                    800, length*ystretch, BufferedImage.TYPE_INT_ARGB
                );
                final Graphics2D imggfx2 = (Graphics2D)img2.getGraphics();
                imggfx2.setRenderingHint(
                    RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_ON
                );
                imggfx2.setStroke(stroke);
                imggfx2.setBackground(Color.WHITE);
                //
                final JPanel canvas2 = new JPanel(){
                    private static final long serialVersionUID = -5927396264951085674L;
                    //
                    @Override
                    protected void paintComponent(Graphics gfx) {
                        super.paintComponent(gfx);
                        gfx.drawImage(img2, 0, 0, null);
                    }
                };
                //
                final Dimension canvasdim2 = new Dimension(
                    img2.getWidth(), img2.getHeight()
                );
                //
                canvas2.setPreferredSize(canvasdim2);
                canvas2.setSize(canvasdim2);
                //
                imggfx2.clearRect(0, 0, img2.getWidth(), img2.getHeight());
                 
                //  
                
                final JFrame frame2 = new JFrame("forgetgate");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                
                //
                frame2.add(canvas2);
                frame2.pack();
                frame.setBounds(0, 0, img.getWidth(), img.getHeight());
                frame2.setBounds(frame.getWidth(), 0, img2.getWidth(), img2.getHeight());
                frame2.setVisible(true);
                
        	    // colors for gatelines
                Color[] colors = new Color[100];
                for(int i=0; i<colors.length;i++){
                	colors[i] = new Color((123*i)%255,(456*i)%255,(789*i)%255); //not random because i want same colors for same cells each run
                }
                          
            net.rebuffer(length);
                final double[] input = testset.get(0).input.data;
                final double[] target = testset.get(0).target.data;
                final double[] netoutput = new double[target.length];
     
            produceNetoutput(net, input,target,netoutput);
            
                final ArrayList<Line> LineListe = getLines(target,netoutput,input);
               
                
                
                
                //prepare combobox for selection of different gates 
                int gatesNum = net.getStructure().arraysnum;
                final int[] selectedGate =  new int[]{2};
                final JComboBox c = new JComboBox();
               
                
                c.addActionListener(new ActionListener() {
          	      public void actionPerformed(ActionEvent e) {
          	    	  selectedGate[0]= c.getSelectedIndex();
          	    	imggfx.clearRect(0, 0, img.getWidth(), img.getHeight());
                    imggfx2.clearRect(0, 0, img.getWidth(), img.getHeight());
                    timestep[0] = 0;
                    frame2.setTitle((String) c.getSelectedItem());
          	      }
          	    });

                frame2.getContentPane().add(c);  
                	  
            
              final double[][][] gates = new double[gatesNum][][];
              final ArrayList<ArrayList<Line>> gateLines = new ArrayList<ArrayList<Line>>();
            for(int i = 0; i< gatesNum; i++){
            	final ArrayList<Line> LineListe2 = new ArrayList<Line>();
                  c.addItem(net.getStructure().arrays[i].tag);
                  
       
                   gates[i] = getGates(net,i);
                   
                   for(int j=0; j<gates[i].length;j++){
                   	LineListe2.add(new Line(colors[j],gates[i][j]));
                   }
                  gateLines.add(LineListe2);
                 }

                
                
             //final Net mlp = trainMLP(net,target,eventarray,trainmlp);
            final Net mlp2 = trainMLP(net,target,eventarray,trainmlp,2);
            final Net mlp1 = trainMLP(net,target,eventarray,trainmlp,1);
             
             frame.addMouseListener(new MouseAdapter() {
                 @Override
                 public void mousePressed(MouseEvent e) {
                     super.mousePressed(e);
                         
                     System.out.println("Vorhersage aufgrund forgetgate:");
                     predictNextEvent(getGate(net,2)[timestep[0]],mlp2,eventarray);
                     System.out.println("Vorhersage aufgrund inputgate:");
                     predictNextEvent(getGate(net,1)[timestep[0]],mlp1,eventarray);
                        // predictNextEvent(net.getOutputBuffer(timestep[0]),mlp,eventarray);
                         
                         
                 }
             });
                
            final Timer timer = new Timer((100), new ActionListener() {
                //
         
                
                
                public void actionPerformed(ActionEvent e) {
                    //
                   
                        //
                	if(timestep[0]>=length){
                		imggfx.clearRect(0, 0, img.getWidth(), img.getHeight());
                        imggfx2.clearRect(0, 0, img.getWidth(), img.getHeight());
                		timestep[0]=0;
                	}
                    
       
                        if (timestep[0] > 0) {
                            //
                        	final int w = img.getWidth();
                            final int h = img.getHeight();
                            final int mx = w / 2;
                            final int my = h / 2;
                            
                        	for(Line l : LineListe){

                        		
                        		if(l.oneDim){
                        			int x1 = mx+(int)(l.data[timestep[0]-1]*0.8*mx);
                        			int x2 = mx+(int)(l.data[timestep[0]]*0.8*mx);
                        			int y1 = (timestep[0]-1)*ystretch;
                        			int y2 = (timestep[0])*ystretch;
                        			
                        			
                        			imggfx.setColor(l.color);
                                    imggfx.drawLine(x1,y1,x2,y2);
                        		}
                        		if(l.event){
                        			
                        			int x1 = mx+(int)(0.8*mx);
                        			int x2 = mx+(int)(0.8*mx);
                        			int y1 = my-(int)(0.8*my);
                        			int y2 = my+(int)(0.8*my);
                        			int a = (int) (l.data[timestep[0]*2]*240+2);
                        			
                        			imggfx.setColor(new Color(a,a,a));
                                    imggfx.drawLine(x1,y1,x2,y2);
                                     x1 = mx-(int)(0.8*mx);
                        			 x2 = mx-(int)(0.8*mx);
                        			y1 = my-(int)(0.8*my);
                        			 y2 = my+(int)(0.8*my);
                        			a = (int) (l.data[timestep[0]*2+1]*240+2);
                        		
                        			imggfx.setColor(new Color(a,a,a));
                                    imggfx.drawLine(x1,y1,x2,y2);
                        		}
                        	}
                        	
                        		for(Line l : gateLines.get(selectedGate[0])){

                        		
                        		if(l.oneDim){
                        			int x1 = mx+(int)(l.data[timestep[0]-1]*0.8*mx);
                        			int x2 = mx+(int)(l.data[timestep[0]]*0.8*mx);
                        			int y1 = (timestep[0]-1)*ystretch;
                        			int y2 = (timestep[0])*ystretch;
                        			
                        	
                        			imggfx2.setColor(l.color);
                                    imggfx2.drawLine(x1,y1,x2,y2);
                        		}
                        	}
                        		for(Event ev : eventarray){
                        			//TODO ball position unabhängig von netoutput. manchmal = output
                        			if(ev.uperbound){
                        				if(target[timestep[0]]<ev.value)
                        					continue;
                        			}else{
                        				if(target[timestep[0]]>ev.value)
                        					continue;
                        			}
                        			int x1 = mx+(int)(-0.8*mx);
                        			int x2 = mx+(int)(0.8*mx);
                        			int y1 = (timestep[0])*ystretch;
                        			int y2 = (timestep[0])*ystretch;
                        			
                        			System.out.println(ev.name);
                        			imggfx2.setColor(ev.color);
                                    imggfx2.drawLine(x1,y1,x2,y2);
                        		}
                        	
                            
                        }
                
                        timestep[0]++;
                    
                    canvas.repaint();
                    canvas2.repaint();
                }
            });
            //
            timer.start();
        
        
      
    

    }
    
    private double produceNetoutputOnline(Net net) {
	
    	if(in.length>1||(in[0]!="pos"&&in[0]!="noicepos")){
    		System.out.println("Feedback & Online not possible");
    		return -1;
    	}
    	int sp = -1;
    	int pn = -1;
    	for(int i=0; i<out.length;i++){
			if(out[i]=="speed") sp=i;
			if(out[i]=="posnext") pn=i;
		}
    	if(sp*pn==1){
    		System.out.println("Feedback & Online not possible");
    		return -1;
    	}
    	double[] speed = new double[inlayer];
    	double[] out = new double[outlayer];
    	
    	for(int j=0; j<inlayer; j++){
    		speed[j]= ballin[j]- ballinprev[j];
    		if(Math.abs(ballin[j])>1){
    			speed[j] *= -1;
    		}
    		ballinprev[j]= ballin[j];
    		ballin[j] = ballinprev[j]+speed[j];
    	}
    	
    	double error =0;
    	
            	
            	
            	
            	
            	net.input(ballinprev, 0);
            	net.compute();
            	net.output(out,  0);
            	if(pn>=0){
            		net.target(ballin, 0); 
            		net.injectError();
                    error += Error.computeRMSE(
                            out, 0, ballin, 0, outlayer );
            	}else{ //output = speed
            		net.target(speed, 0);
            		net.injectError();
                    error += Error.computeRMSE(
                            out, 0, speed, 0, outlayer );
            	}
            	if(feedback){
            		if(ballin.length != out.length) while(true)System.out.println("Online für multiplen Output nicht fertig");
            		for(int j=0; j<inlayer; j++){
            			if(pn>=0)
            				ballin[j]=out[j];
            			else
            				ballin[j]= ballinprev[j]+speed[j];
            		}
            	}
            
    
		return error;
	}

	private void generateOnlineSample(Net net, Sample sample) {
		// include washout for net
		net.reset();
		int teacher = 100;
		while (teacher*inlayer > sample.input.data.length) teacher -= 10; //should never happen
		for(int i=0; i <teacher;i++){
		
    	for(int j=0;j<inlayer;j++){
    		ballinprev[j] = ballin[j];
    		ballin[j]=sample.input.data[i*inlayer+j];
    	}
    	net.input(ballin, 0);
    	net.compute();
		}
	}
    
    private double produceNetoutput(Net net, double[] input, double[] target, double[] netoutput) {
		
    	
    	if(in.length>1||(in[0]!="pos"&&in[0]!="noicepos")){
    		System.out.println("Feedback not possible");
    		feedback = false;
    	}
    	int sp = -1;
    	int pn = -1;
    	for(int i=0; i<out.length;i++){
			if(out[i]=="speed") sp=i;
			if(out[i]=="posnext") pn=i;
		}
    	if(sp*pn==1){
    		System.out.println("Feedback not possible");
    		feedback = false;
    	}
    	double[] in = new double[inlayer];
    	double[] out = new double[outlayer];
    	double[] tar = new double[outlayer];
    	double error =0;
    	if(feedback){
    		int teacher = (int) (length*0.3);
    		
            net.reset();
            for(int i=0; i<input.length/inlayer;i++){
            if(i<teacher){
            	for(int j=0;j<inlayer;j++)
            		in[j]=input[i*inlayer+j];
            	net.setFrameIdx(i);
            	net.input(in, 0);
            	net.compute();
            	net.output(out,  0);
            	for(int j=0;j<outlayer;j++)
            		tar[j]=target[i*outlayer+j];
            	net.target(tar, 0);
                net.injectError();
                error += Error.computeRMSE(
                        out, 0, tar, 0, outlayer );
            	for(int j=0;j<outlayer;j++)
            		netoutput[i*outlayer+j]=out[j];
            }else{
            	if(pn>=0){
            	for(int j=0;j<inlayer;j++){
            		in[j]=out[pn+j];
            		input[i*inlayer+j]=in[j];
            		}
            	}else{
            	for(int j=0;j<inlayer;j++){
                	in[j]+=out[sp+j];
                	input[i*inlayer+j]=in[j];
            		}
            	}
            	net.setFrameIdx(i);
            	net.input(in, 0);
            	net.compute();
            	net.output(out,  0);
            	for(int j=0;j<outlayer;j++)
            		tar[j]=target[i*outlayer+j];
            	net.target(tar, 0);
                net.injectError();
                error += Error.computeRMSE(
                        out, 0, tar, 0, outlayer );
            	for(int j=0;j<outlayer;j++)
            		netoutput[i*outlayer+j]=out[j];
            }
            	
            }
    	}else{

        net.reset();
        for(int i=0; i<input.length/inlayer;i++){
        	for(int j=0;j<inlayer;j++)
        		in[j]=input[i*inlayer+j];
        	net.setFrameIdx(i);
        	net.input(in, 0);
        	net.compute();
        	net.output(out,  0);
        	for(int j=0;j<outlayer;j++)
        		tar[j]=target[i*outlayer+j];
        	net.target(tar, 0);
            net.injectError();
            error += Error.computeRMSE(
                    out, 0, tar, 0, outlayer );
        	for(int j=0;j<outlayer;j++)
        		netoutput[i*outlayer+j]=out[j];
        }
      }
		return error;
	}

    
	private ArrayList<Line> getLines(double[] target, double[] netoutput, double[] input) {
		ArrayList<Line> LineListe = new ArrayList<Line>();
		double[] data = new double[input.length/inlayer];
        for(int i=0; i<in.length; i++){
        	if(in[i]=="zero"){
        		for(int t=0;t<data.length;t++)
        		data[t]=input[t*inlayer+i];
        		LineListe.add(new Line(new Color(80, 220, 20),data));
        	}
        	if(in[i]=="pos"){
        		for(int t=0;t<data.length;t++)
        		data[t]=input[t*inlayer+i];
        		LineListe.add(new Line(new Color(80, 220, 20),data));
        	}
        	if(in[i]=="noicepos"){
        		for(int t=0;t<data.length;t++)
        		data[t]=input[t*inlayer+i];
        		LineListe.add(new Line(new Color(80, 220, 20),data));
        	}
        	if(in[i]=="speed"){
        		for(int t=0;t<data.length;t++)
            		data[t]=input[t*inlayer+i];
            		LineListe.add(new Line(new Color(80, 220, 20),data));
        	}
        }
		
        double[] data2 = new double[target.length/outlayer];
        double[] data3 = new double[target.length/outlayer*eventarray.length];
        for(int i=0; i<out.length; i++){
        	if(out[i]=="pos"){
        		for(int t=0;t<data2.length;t++)
        			data2[t]=target[t*outlayer+i];
            	LineListe.add(new Line(new Color(20, 80, 220),data2.clone()));
            	for(int t=0;t<data2.length;t++)
            		data2[t]=netoutput[t*outlayer+i];
                LineListe.add(new Line(new Color(220, 80, 20),data2.clone()));
        	}
        	if(out[i]=="posnext"){
        		for(int t=0;t<data2.length;t++)
        			data2[t]=target[t*outlayer+i];
               	LineListe.add(new Line(new Color(20, 80, 220),data2.clone()));
               	for(int t=0;t<data2.length;t++)
               		data2[t]=netoutput[t*outlayer+i];
                LineListe.add(new Line(new Color(220, 80, 20),data2.clone()));
        	}
        	if(out[i]=="speed"){
        		for(int t=0;t<data2.length;t++)
        			data2[t]=target[t*outlayer+i];
                LineListe.add(new Line(new Color(20, 80, 220),data2.clone()));
                for(int t=0;t<data2.length;t++)
                	data2[t]=netoutput[t*outlayer+i];
                LineListe.add(new Line(new Color(220, 80, 20),data2.clone()));
        	}
        	if(out[i]=="event"){
        		for(int j=0;j<eventarray.length;j++){
        			for(int t=0;t<data2.length;t++)
        				data3[t*2+j]=target[t*outlayer+i+j];
        		}// end for each event
            //         LineListe.add(new Line(new Color(20, 80, 220),data3.clone(),false,true));
        		for(int j=0;j<eventarray.length;j++){
                     for(int t=0;t<data2.length;t++)
                    	 data3[t*2+j]=netoutput[t*outlayer+i+j];
        		}//end for each event
                     LineListe.add(new Line(new Color(220, 80, 20),data3.clone(),false,true));
        		
        }// end if event should go to out
        }// end each output

        
		return LineListe;
	}



	protected static void predictNextEvent(double[] ds, Net mlp, Event[] eventarray) {
		
		double[] out = new double[eventarray.length];
		mlp.reset();
        mlp.input(ds, 0);
        mlp.compute();
        mlp.output(out,  0);
        
        
        for(int i =0; i< eventarray.length;i++){
        	System.out.println(eventarray[i].name + ": "+ new DecimalFormat("##.##").format(out[i]*100)+"%");
        }
		
		
	}



	private  Net trainMLP(Net net, double[] target, Event[] eventarray, boolean trainmlp) throws IOException, ClassNotFoundException {
		 //TODO macht nur sinn wenn target position ist 
		// 
		MLPGenerator gen = new MLPGenerator();
	        //
	        // setup layers.
	        //
		 
		    gen.inputLayer(net.getOutputBuffer(0).length);
	        gen.hiddenLayer(2, CellType.TANH);
	        gen.outputLayer(eventarray.length, CellType.SIGMOID, true, -1.0);
	        //
	        // just generate.
	        //
	        Net mlp = gen.generate(); 
	        if(trainmlp){
		 
		 	SampleSet set = new SampleSet();
		 	int sincelastevent = 0;
		 	
		 	for(int t=0; t<target.length;t++){
		 		int eventnum = 0;
		 		for(Event ev : eventarray){
        			if(ev.uperbound){
        				if(target[t]<ev.value){
        					eventnum++;
        					continue;
        				}
        			}else{
        				if(target[t]>ev.value){
        					eventnum++;
        					continue;
        				}
        			}
        			//event ist ausgelöst
        			double[] targetmlp = new double[eventarray.length];
        			for(int j=0;j<eventarray.length;j++){
        				if(j==eventnum)
        					targetmlp[j]=1;
        				else
        					targetmlp[j]=0;
        			}
        			for(int i=sincelastevent;i<=t;i++){
        				set.add(new Sample(net.getOutputBuffer(i),targetmlp));
        			}
        			sincelastevent = t;
        			
        			
		 	 } // end for each event
		 	} // for each t
		 	
		 	
	       
	        
	        GradientDescent optimizer = new GradientDescent();
	        optimizer.setLearningRate(0.2);
	        optimizer.setRnd(rnd);
	        optimizer.setParameters(mlp.getWeightsNum());
	        optimizer.setMomentum(0.5);
	        //
	        // setup learning.
	        //
	        NetLearning learning = new NetLearning();
	        learning.addListener(new BasicNetLearningListener());
	        learning.setRnd(rnd);
	        learning.setNet(mlp);
	        learning.setSampling(Sampling.STOCHASTIC);
	        learning.setTrainingSamples(set);
	        learning.setEpochs(5);
	        learning.setOptimizer(optimizer);
	        //
	        // perform training and print final error.
	        //
	        learning.learn();
	        
	        final double[] weights = new double[mlp.getWeightsNum()];
            mlp.readWeights(weights, 0);
            Serializer.write(weights, weightString+"MLP.weights");
		 	
	        }else{
	        	final double[] weights = Serializer.read(weightString+"MLP.weights");
	            mlp.writeWeights(weights, 0);
	        	
	        }
	 
	        return mlp;
	}
	private  Net trainMLP(Net net, double[] target, Event[] eventarray, boolean trainmlp, int gate) throws IOException, ClassNotFoundException {
		 MLPGenerator gen = new MLPGenerator();
	        //
	        // setup layers.
	        //
		    double[][] gateact =getGate(net,gate);
		    gen.inputLayer(gateact[0].length);
	        gen.hiddenLayer(2, CellType.TANH);
	        gen.outputLayer(eventarray.length, CellType.SIGMOID, true, -1.0);
	        //
	        // just generate.
	        //
	        Net mlp = gen.generate(); 
	        if(trainmlp){
		 
		 	SampleSet set = new SampleSet();
		 	int sincelastevent = 0;
		 	for(int t=0; t<target.length/outlayer;t++){
		 		int eventnum = 0;
		 		for(Event ev : eventarray){
       			if(ev.uperbound){
       				if(target[t]<ev.value){
       					eventnum++;
       					continue;
       				}
       			}else{
       				if(target[t]>ev.value){
       					eventnum++;
       					continue;
       				}
       			}
       			//event ist ausgelöst
       			double[] targetmlp = new double[eventarray.length];
       			for(int j=0;j<eventarray.length;j++){
       				if(j==eventnum)
       					targetmlp[j]=1;
       				else
       					targetmlp[j]=0;
       			}
       			for(int i=sincelastevent;i<=t;i++){
       				set.add(new Sample(gateact[i],targetmlp));
       			}
       			sincelastevent = t;
       			
       			
		 	 } // end for each event
		 	} // for each t
		 	
		 	
	       
	        
	        GradientDescent optimizer = new GradientDescent();
	        optimizer.setLearningRate(0.2);
	        optimizer.setRnd(rnd);
	        optimizer.setParameters(mlp.getWeightsNum());
	        optimizer.setMomentum(0.5);
	        //
	        // setup learning.
	        //
	        NetLearning learning = new NetLearning();
	        learning.addListener(new BasicNetLearningListener());
	        learning.setRnd(rnd);
	        learning.setNet(mlp);
	        learning.setSampling(Sampling.STOCHASTIC);
	        learning.setTrainingSamples(set);
	        learning.setEpochs(5);
	        learning.setOptimizer(optimizer);
	        //
	        // perform training and print final error.
	        //
	        learning.learn();
	        
	        final double[] weights = new double[mlp.getWeightsNum()];
           mlp.readWeights(weights, 0);
           Serializer.write(weights, weightString+"MLP"+gate+".weights");
		 	
	        }else{
	        	final double[] weights = Serializer.read(weightString+"MLP"+gate+".weights");
	            mlp.writeWeights(weights, 0);
	        	
	        }
	 
	        return mlp;
	}



	private static double[][] getGates(Net net, int g) {
		final int idx = net.getStructure().arrays[g].cellslbd;
        final int num = net.getStructure().arrays[g].cellsnum;
      
        double[][] act =new double [num][net.getFramesNum()];
       for(int j=0; j<net.getFramesNum();j++){
    		net.setFrameIdx(j);
        for (int i = idx; i < num+idx; i++) {
              act[i-idx][j] = net.getOutputBuffer(j)[i];
        }
       }
       
		return act;
	}
	private static double[][] getGate(Net net, int g) { // andersrum als darüber, achtung!
		final int idx = net.getStructure().arrays[g].cellslbd;
        final int num = net.getStructure().arrays[g].cellsnum;
      
        double[][] act =new double [net.getFramesNum()][num];
       for(int j=0; j<net.getFramesNum();j++){
    		net.setFrameIdx(j);
        for (int i = idx; i < num+idx; i++) {
              act[j][i-idx] = net.getOutputBuffer(j)[i];
        }
       }
       
		return act;
	}
	

       
      
    
    
    
}
