
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
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
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.Timer;

import de.jannlab.Net;
import de.jannlab.core.CellType;
import de.jannlab.data.Sample;
import de.jannlab.data.SampleSet;
import de.jannlab.error.Error;
import de.jannlab.generator.GenerateNetworks;
import de.jannlab.generator.LSTMGenerator;
import de.jannlab.generator.MLPGenerator;
import de.jannlab.io.Serializer;
import de.jannlab.learning.BasicNetLearningListener;
import de.jannlab.learning.NetLearning;
import de.jannlab.learning.Sampling;
import de.jannlab.misc.TimeCounter;
import de.jannlab.optimization.BasicOptimizationListener;
import de.jannlab.optimization.DifferentiableObjective;
import de.jannlab.optimization.optimizer.Adam;
import de.jannlab.optimization.optimizer.GradientDescent;


public class BouncingBall2D {
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
	private boolean feedback;
	private String weightString;
	private int currepoch;
	
    private static TimeCounter TC = new TimeCounter();
    private static Random rnd = new Random(0L);
    
   
    
    public BouncingBall2D(boolean random, String[] in, String[] out, String hid, boolean train, boolean trainmlp, int epochs, int length, int trainsize, double learningrate, double beta1, double beta2, double epsilon, boolean biascorrection, int ystretch, boolean feedback) throws ClassNotFoundException, IOException{
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
        this.currepoch = 0;
        this.eventarray = new Event[]{new Event("Linker Bounce",0,-1,false,Color.gray), new Event("Rechter Bounce",0,1,true,Color.black),new Event("Unterer Bounce",1,1,true,Color.black),new Event("Oberer Bounce",1,-1,false,Color.black)};
        inlayer=0;
        for(int i=0; i<in.length; i++){
        	if(in[i]=="zero"){
        		inlayer +=2;
        	}
        	if(in[i]=="pos"){
        		inlayer +=2;
        	}
        	if(in[i]=="noicepos"){
        		inlayer +=2;
        	}
        	if(in[i]=="speed"){
        		inlayer +=2;
        	}
        }
        outlayer = 0;
        for(int i=0; i<out.length; i++){
        	if(out[i]=="pos"){
        		outlayer +=2;
        	}
        	if(out[i]=="posnext"){
        		outlayer +=2;
        	}
        	if(out[i]=="speed"){
        		outlayer +=2;
        	}
        	if(out[i]=="event"){
        		outlayer +=eventarray.length;
        	}
        }
        run();
	}//end constructor



	public  Sample generateSample(final int length) {
    	
    	
    	
    	double[] start = new double []{0.9,0.1};
    	double[] speed = new double []{-0.05,0.05};
    	if(random){
    		start[0]+=Math.random()*0.1-0.05;
    		start[1]+=Math.random()*0.1-0.05; 
    		speed[0]+=Math.random()*0.02-0.01; //0.04-0.06
    		speed[1]+=Math.random()*0.02-0.01;
    		if(Math.random()>0.5){
    			speed[0]*=-1;
    			speed[1]*=-1;
    		}
    		/*
    	 start= new double []{Math.random()*1.8-0.9,Math.random()*1.8-0.9};
         speed = new double []{Math.random()*0.2-0.1,Math.random()*0.2-0.1};
         */
    	}
    	double inputnoice = 0.01;
    	int evnum = eventarray.length;
    	
    	//generieren der daten
    	double[] pos = new double[length*2];
    	double[] noicepos = new double[length*2];
        double[] speeds =  new double[length*2];
        double[] events = new double[length*evnum];
        
	 	int sincelastevent = 0;
	 	for(int t=0; t<length;t++){
	 		if(t==0){ pos[0]= start[0];  pos[1]= start[1];}
	 		else{ pos[t*2]= pos[(t-1)*2]+speed[0];
	 		pos[t*2+1]= pos[(t-1)*2+1]+speed[1];
	 		}
	 		noicepos[t*2] = pos[t*2]+(Math.random()*2-1)*inputnoice;
	 		noicepos[t*2+1] = pos[t*2+1]+(Math.random()*2-1)*inputnoice;
	 		speeds[t*2]=speed[0];
	 		speeds[t*2+1]=speed[1];
	 		if(Math.abs(pos[t*2])>=1){
             	speed[0] *=-1;
             	
             }
			if(Math.abs(pos[t*2+1])>=1){
             	speed[1] *=-1;
             }
			
			int eventnum = 0;
	 		for(Event ev : eventarray){
	 			if(ev.uperbound){
    				if(pos[t*2+ev.varnum]<ev.value){
    					eventnum++;
    					continue;
    				}
    			}else{
    				if(pos[t*2+ev.varnum]>ev.value){
    					eventnum++;
    					continue;
    				}
    			}
    			//event ist ausgelöst
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
            		data[t*inlayer+i*2]=0;
            		data[t*inlayer+i*2+1]=0;
            	}
            	if(in[i]=="pos"){
            		data[t*inlayer+i*2]=pos[t*2];
            		data[t*inlayer+i*2+1]=pos[t*2+1];
            	}
            	if(in[i]=="noicepos"){
            		data[t*inlayer+i*2]=noicepos[t*2];
            		data[t*inlayer+i*2+1]=noicepos[t*2+1];
            	}
            	if(in[i]=="speed"){
            		data[t*inlayer+i*2]=speeds[t*2];
            		data[t*inlayer+i*2+1]=speeds[t*2+1];
            	}
            }
           
            for(int i=0; i<out.length; i++){
            	if(out[i]=="pos"){
            		target[t*outlayer+i*2]=pos[t*2];
            		target[t*outlayer+i*2+1]=pos[t*2+1];
            	}
            	if(out[i]=="posnext"){
            		if(t==length-1){
            		target[t*outlayer+i*2]=pos[t*2]+speeds[t*2];
            		target[t*outlayer+i*2+1]=pos[t*2+1]+speeds[t*2+1];
            		}else{
            		target[t*outlayer+i*2]=pos[(t+1)*2];
            		target[t*outlayer+i*2+1]=pos[(t+1)*2+1];
            		}
            	}
            	if(out[i]=="speed"){
            		target[t*outlayer+i*2]=speeds[t*2];
            		target[t*outlayer+i*2+1]=speeds[t*2+1];
            	}
            	if(out[i]=="event"){
            		for(int j=0;j<evnum;j++){
            	target[t*outlayer+i*2+j]=events[t*evnum+j];
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
    	
    	weightString = "2DIn";
        
        for(int i=0; i<in.length; i++){
        	weightString +=in[i];
        }
        weightString +=hid+"out";
        for(int i=0; i<out.length; i++){
        	weightString +=out[i];
        }
        weightString +="random"+random;
        String netString = "LSTM-"+inlayer+hid+"linear"+outlayer+"b";
        final Net net = GenerateNetworks.generateNet(netString);
        net.rebuffer(length);
          

          final SampleSet trainset = generate(trainsize, length);
          SampleSet testset  = generate(1, length);
          
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
                    trainset.shuffle(rnd);
                	//washoutphase in gradienten mit rein?
                    double error=0;
                    for(int i=0; i<trainset.size();i++){
                	final double[] input = trainset.get(i).input.data;
                  final double[] target = trainset.get(i).target.data;
                  final double[] netoutput = new double[target.length];
                  
                 error += produceNetoutput(net, input,target,netoutput);
                    }
                	
                	
                	net.computeGradient();
                  net.readGradWeights(grad, gradoffset);
                    currepoch++;
                    return error;
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
  				// TODO Auto-generated method stub
  				return 0;
  			}
  			
            };
            
            //
            // setup trainer.
            //
            File f = new File(weightString+".weights");
      		if(f.exists() && !f.isDirectory()) { 
      			final double[] weights = Serializer.read(weightString+".weights");
                net.writeWeights(weights, 0);
      		}else{
      			currepoch=1;
      		}
            
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
          
      	
      	final int[] timestep = {0};
      	
        final BufferedImage img = new BufferedImage(
                650, 650, BufferedImage.TYPE_INT_ARGB
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
             
            //  
            
            final JFrame frame = new JFrame("2D Bouncing Ball!. " + epochs + " Epochen.");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            //
            frame.add(canvas);
            frame.pack();
            frame.setVisible(true);
            
            //second view
            
            final BufferedImage img2 = new BufferedImage(
                    650, 650, BufferedImage.TYPE_INT_ARGB
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
                
                final JFrame frame2 = new JFrame("2D Bouncing Ball!. " + epochs + " Epochen.");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame2.add(canvas2);
                frame2.pack();
                frame2.setVisible(true);
                
                // 3rt view
                final BufferedImage img3 = new BufferedImage(
                        650, length*ystretch, BufferedImage.TYPE_INT_ARGB
                    );
                    final Graphics2D imggfx3 = (Graphics2D)img3.getGraphics();
                    imggfx3.setRenderingHint(
                        RenderingHints.KEY_ANTIALIASING,
                        RenderingHints.VALUE_ANTIALIAS_ON
                    );
                    imggfx3.setStroke(stroke);
                    imggfx3.setBackground(Color.WHITE);
                    //
                    final JPanel canvas3 = new JPanel(){
                        private static final long serialVersionUID = -5937396364951085674L;
                        //
                        @Override
                        protected void paintComponent(Graphics gfx) {
                            super.paintComponent(gfx);
                            gfx.drawImage(img3, 0, 0, null);
                        }
                    };
                    //
                    final Dimension canvasdim3 = new Dimension(
                        img3.getWidth(), img3.getHeight()
                    );
                    //
                    canvas3.setPreferredSize(canvasdim3);
                    canvas3.setSize(canvasdim3);
                    //
                    imggfx3.clearRect(0, 0, img3.getWidth(), img3.getHeight());
                     
                    //  
                    
                    final JFrame frame3 = new JFrame("outputgate");
                    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                    frame3.add(canvas3);
                    frame3.pack();
                    frame.setBounds(0, 0, img.getWidth(), img.getHeight());
                    frame.setBounds(0, 0, img.getWidth(), img.getHeight());
                    frame2.setBounds(frame.getWidth(), 0, img2.getWidth(), img2.getHeight());
                    frame3.setBounds(frame.getWidth()+frame2.getWidth(), 0, img3.getWidth(), img3.getHeight());
                    frame3.setVisible(true);
                

            final double[] input = testset.get(0).input.data;
            final double[] target = testset.get(0).target.data;
            final double[] netoutput = new double[target.length];
            
            produceNetoutput(net, input,target,netoutput);

                
                final ArrayList<Line> LineListe = getLines(target,netoutput,input);
                
             // colors for gatelines
                Color[] colors = new Color[100];
                for(int i=0; i<colors.length;i++){
                	colors[i] = new Color((123*i)%255,(456*i)%255,(789*i)%255); //not random because i want same colors for same cells each run
                }
                
                //prepare combobox for selection of different gates 
                int gatesNum = net.getStructure().arraysnum;
                final int[] selectedGate =  new int[]{8};
                final JComboBox c = new JComboBox();
               
                
                c.addActionListener(new ActionListener() {
          	      public void actionPerformed(ActionEvent e) {
          	    	  selectedGate[0]= c.getSelectedIndex();
          	    	imggfx.clearRect(0, 0, img.getWidth(), img.getHeight());
                    imggfx2.clearRect(0, 0, img2.getWidth(), img2.getHeight());
                    imggfx3.clearRect(0, 0, img3.getWidth(), img3.getHeight());
                    timestep[0] = 0;
                    frame3.setTitle((String) c.getSelectedItem());
          	      }
          	    });

                frame3.getContentPane().add(c);  
                	  
            
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
                        
                        //
                        System.out.println("Vorhersage aufgrund forgetgate:");
                        predictNextEvent(getGate(net,2)[timestep[0]],mlp2,eventarray);
                        System.out.println("Vorhersage aufgrund inputgate:");
                        predictNextEvent(getGate(net,1)[timestep[0]],mlp1,eventarray);
                       // predictNextEvent(net.getOutputBuffer(timestep[0]),mlp,eventarray);
                        
                        
                }
            });
    
    
    //
    // setup render timer.
    //
            
                
            final Timer timer = new Timer((80), new ActionListener() {
                //
      
                
                public void actionPerformed(ActionEvent e) {
                    //
                   
                        //
                	if(timestep[0]>=length){
                		timestep[0]=0;
                		imggfx.clearRect(0, 0, img.getWidth(), img.getHeight());
                		imggfx2.clearRect(0, 0, img2.getWidth(), img2.getHeight());
                		imggfx3.clearRect(0, 0, img3.getWidth(), img3.getHeight());
                    	
                	}
                        
                      
                        //
                        if (timestep[0] > 0) {
                            //
                        	imggfx2.clearRect(0, 0, img2.getWidth(), img2.getHeight());
                        	
                            final int w = img.getWidth();
                            final int h = img.getHeight();
                            final int mx = w / 2;
                            final int my = h / 2;
                            
                       	for(Line l : LineListe){

                       		imggfx.setColor(l.color);
                       		imggfx2.setColor(l.color);
                       		
                        		if(l.oneDim){
                        			int x1 = mx+(int)(l.data[timestep[0]-1]*0.8*mx);
                        			int x2 = mx+(int)(l.data[timestep[0]]*0.8*mx);
                        			int y1 = (timestep[0]-1)*ystretch;
                        			int y2 = (timestep[0])*ystretch;

                                    imggfx.drawLine(x1,y1,x2,y2);
                        		}else{              			
                            		if(l.event){
                            			int x1,x2,y1,y2;
                            			for(int i=0; i<eventarray.length;i++){
                            			 int a = (int) (Math.abs(1-l.data[timestep[0]*4+i])*255);
                            				if(eventarray[i].varnum==0){
                            					 x1 = mx+ eventarray[i].value * (int)(0.8*mx);
                                       			 x2 = mx+ eventarray[i].value * (int)(0.8*mx);
                                       			 y1 = my-(int)(0.8*my);
                                       			 y2 = my+(int)(0.8*my);
                            				}else{
                            					 y1 = my+ eventarray[i].value * (int)(0.8*my);
                                       			 y2 = my+ eventarray[i].value * (int)(0.8*my);
                                       			 x1 = mx-(int)(0.8*mx);
                                       			 x2 = mx+(int)(0.8*mx);
                            				}
                            				imggfx2.setColor(new Color(a,a,a));
                                            imggfx2.drawLine(x1,y1,x2,y2);
                            			} // for each event

                            		}else{
                        			
                        			int x1 = mx+(int)(l.data[(timestep[0]-1)*2]*0.8*mx);
                        			int x2 = mx+(int)(l.data[(timestep[0])*2]*0.8*mx);
                        			int y1 = my+(int)(l.data[(timestep[0]-1)*2+1]*0.8*my);
                        			int y2 = my+(int)(l.data[(timestep[0])*2+1]*0.8*my);
                        			imggfx.drawLine(x1,y1,x2,y2);
                        			imggfx2.fillOval(x2, y2, 9, 9);
                            		}}
                        	}
                 		for(Line l : gateLines.get(selectedGate[0])){

                    		
                    		if(l.oneDim){
                    			int x1 = mx+(int)(l.data[(timestep[0]-1)]*0.8*mx);
                    			int x2 = mx+(int)(l.data[timestep[0]]*0.8*mx);
                    			int y1 = (timestep[0]-1)*ystretch;
                    			int y2 = (timestep[0])*ystretch;
                    			
                    	
                    			imggfx3.setColor(l.color);
                                imggfx3.drawLine(x1,y1,x2,y2);
                    		}
                    	}
                 		if(in[0]=="pos"){
                 		for(Event ev : eventarray){
                			if(ev.uperbound){
                				if(input[timestep[0]*inlayer+ev.varnum]<ev.value)
                					continue;
                			}else{
                				if(input[timestep[0]*inlayer+ev.varnum]>ev.value)
                					continue;
                			}
                			//event ausgelöst
                			int x1 = mx+(int)(-0.9*mx);
                			int x2 = mx+(int)(0.9*mx);
                			int y1 = (timestep[0])*ystretch;
                			int y2 = (timestep[0])*ystretch;
                			
                			System.out.println(ev.name);
                			imggfx3.setColor(ev.color);
                            imggfx3.drawLine(x1,y1,x2,y2);
                		}
                 		}
                         
                            
                        }
                       
                        
                        //
                
                        timestep[0]++;
                    
                    canvas.repaint();
                    canvas2.repaint();
                    canvas3.repaint();
                    
                }
            });
            //
            timer.start();
        
        
      

    }
    
    private double produceNetoutput(Net net, double[] input, double[] target, double[] netoutput) {
		// TODO Auto-generated method stub
    	
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
    	double error=0;
    	if(feedback){
    		//int teacher = (int) (length*(0.3+1-((float)currepoch)/epochs)); 
    		int teacher = (int) (length*0.7);

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
            		in[j]=out[pn*2+j];
            		input[i*inlayer+j]=in[j];
            		}
            	}else{
            	for(int j=0;j<inlayer;j++){
                	in[j]+=out[sp*2+j];
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
		double[] data = new double[input.length/inlayer*2];
        for(int i=0; i<in.length; i++){
        	if(in[i]=="zero"){
        		for(int t=0;t<data.length/2;t++){
        		data[t*2]=input[t*inlayer+i*2];
        		data[t*2+1]=input[t*inlayer+i*2+1];
        		}
        		LineListe.add(new Line(new Color(80, 220, 20),data.clone(),false));
        	}
        	if(in[i]=="pos"){
        		for(int t=0;t<data.length/2;t++){
            		data[t*2]=input[t*inlayer+i*2];
            		data[t*2+1]=input[t*inlayer+i*2+1];
            		}
        		LineListe.add(new Line(new Color(80, 220, 20),data.clone(),false));
        	}
        	if(in[i]=="noicepos"){
        		for(int t=0;t<data.length/2;t++){
            		data[t*2]=input[t*inlayer+i*2];
            		data[t*2+1]=input[t*inlayer+i*2+1];
            		}
        		LineListe.add(new Line(new Color(80, 220, 20),data.clone(),false));
        	}
        	if(in[i]=="speed"){
        		for(int t=0;t<data.length/2;t++){
            		data[t*2]=input[t*inlayer+i*2];
            		data[t*2+1]=input[t*inlayer+i*2+1];
            		}
            		LineListe.add(new Line(new Color(80, 220, 20),data.clone(),false));
        	}
        }
		
        double[] data2 = new double[target.length/outlayer*2];
        double[] data3 = new double[target.length/outlayer*eventarray.length];
        for(int i=0; i<out.length; i++){
        	if(out[i]=="pos"){
        		for(int t=0;t<data2.length/2;t++){
        			data2[t*2]=target[t*outlayer+i*2];
        			data2[t*2+1]=target[t*outlayer+i*2+1];
        		}
            	LineListe.add(new Line(new Color(20, 80, 220),data2.clone(),false));
            	for(int t=0;t<data2.length/2;t++){
            		data2[t*2]=netoutput[t*outlayer+i*2];
            	data2[t*2+1]=netoutput[t*outlayer+i*2+1];
        	}
                LineListe.add(new Line(new Color(220, 80, 20),data2.clone(),false));
        	}
        	if(out[i]=="posnext"){
        		for(int t=0;t<data2.length/2;t++){
        			data2[t*2]=target[t*outlayer+i*2];
        			data2[t*2+1]=target[t*outlayer+i*2+1];
        		}
            	LineListe.add(new Line(new Color(20, 80, 220),data2.clone(),false));
            	for(int t=0;t<data2.length/2;t++){
            		data2[t*2]=netoutput[t*outlayer+i*2];
            	data2[t*2+1]=netoutput[t*outlayer+i*2+1];
        	}
                LineListe.add(new Line(new Color(220, 80, 20),data2.clone(),false));
        	}
        	if(out[i]=="speed"){
        		for(int t=0;t<data2.length/2;t++){
        			data2[t*2]=target[t*outlayer+i*2];
        			data2[t*2+1]=target[t*outlayer+i*2+1];
        		}
            	LineListe.add(new Line(new Color(20, 80, 220),data2.clone(),false));
            	for(int t=0;t<data2.length/2;t++){
            		data2[t*2]=netoutput[t*outlayer+i*2];
            	data2[t*2+1]=netoutput[t*outlayer+i*2+1];
        	}
                LineListe.add(new Line(new Color(220, 80, 20),data2.clone(),false));
        	}
        	if(out[i]=="event"){
        		for(int j=0;j<eventarray.length;j++)
        			for(int t=0;t<data2.length/2;t++)
        				data3[t*4+j]=target[t*outlayer+i*2+j];
              //       LineListe.add(new Line(new Color(20, 80, 220),data3.clone(),false,true)); dont show target
                     for(int j=0;j<eventarray.length;j++)
                     for(int t=0;t<data2.length/2;t++)
                    	 data3[t*4+j]=netoutput[t*outlayer+i*2+j];
                    LineListe.add(new Line(new Color(220, 80, 20),data3.clone(),false,true));
        		
        }// end if event should go to out
        }// end each output

        
		return LineListe;
	}
    
	protected static void predictNextEvent(double[] ds, Net mlp, Event[] eventarray) {
		// TODO Auto-generated method stub
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
		 MLPGenerator gen = new MLPGenerator();
	        //
	        // setup layers.
	        //
		 
		    gen.inputLayer(net.getOutputBuffer(0).length);
	        gen.hiddenLayer(6, CellType.TANH);
	        
	       
	        gen.outputLayer(eventarray.length, CellType.SIGMOID, true, -1.0);
	        //
	        // just generate.
	        //
	        Net mlp = gen.generate(); 
	        if(trainmlp){
		 
		 	SampleSet set = new SampleSet();
		 	int sincelastevent = 0;
		 	
		 	for(int t=0; t<target.length/2;t++){
		 		int eventnum = 0;
		 		for(Event ev : eventarray){
       			if(ev.uperbound){
       				if(target[t*2+ev.varnum]<ev.value){
       					eventnum++;
       					continue;
       				}
       			}else{
       				if(target[t*2+ev.varnum]>ev.value){
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
		 	
		 	
	       
	        
	          final Adam optimizer = new Adam();
	          optimizer.setLearningRate(0.001);;
	          optimizer.setBeta1(0.9);
	          optimizer.setBeta2(0.99);
	          optimizer.setEpsilon(1e-8);
	          optimizer.setBiasCorrection(true);
	          optimizer.setRnd(rnd);
	          optimizer.setParameters(mlp.getWeightsNum());
	          //
	          // setup learning.
	          //
	          final NetLearning learning = new NetLearning();
	          learning.addListener(new BasicNetLearningListener());
	          learning.setRnd(rnd);
	          learning.setNet(mlp);
	          learning.setSampling(Sampling.STOCHASTIC);
	          learning.setTrainingSamples(set);
	          learning.setEpochs(50);
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
	        gen.hiddenLayer(6, CellType.TANH);
	        
	       
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
		 		for(Event ev : eventarray){ //only makes sense when only position is in target
      			if(ev.uperbound){
      				if(target[t*2+ev.varnum]<ev.value){
      					eventnum++;
      					continue;
      				}
      			}else{
      				if(target[t*2+ev.varnum]>ev.value){
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
		 	
		 	
	       
	        
	          final Adam optimizer = new Adam();
	          optimizer.setLearningRate(0.001);;
	          optimizer.setBeta1(0.9);
	          optimizer.setBeta2(0.99);
	          optimizer.setEpsilon(1e-8);
	          optimizer.setBiasCorrection(true);
	          optimizer.setRnd(rnd);
	          optimizer.setParameters(mlp.getWeightsNum());
	          //
	          // setup learning.
	          //
	          final NetLearning learning = new NetLearning();
	          learning.addListener(new BasicNetLearningListener());
	          learning.setRnd(rnd);
	          learning.setNet(mlp);
	          learning.setSampling(Sampling.STOCHASTIC);
	          learning.setTrainingSamples(set);
	          learning.setEpochs(50);
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
