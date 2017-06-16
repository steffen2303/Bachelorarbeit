import java.io.IOException;

public class Launcher {

	  public static void main(String[] args) throws IOException, ClassNotFoundException {
	        //
		  
		  	int Dim = 2; 
		  	boolean random = true;
		  	
		  	
		 // zero pos posnext noicepos speed event - keep order!
		  	String in[] = new String[]{"noicepos"}; 
		  	String out[] = new String[]{"speed"};
		  	
		  	String hid = "-tanh6b-";
		  	
	    	boolean train = true;
	    	boolean trainmlp = true; //TODO MLP macht nur Sinn wenn target position ist
	 
	    	
	    	// trainingsparameter
	    	
	        final int    epochs          = 5;
	        final int  length = 300;
	        final int trainsize = 10;
	        final double learningrate    = 0.001;
	        
	        
	        final double beta1           = 0.9; // mt=b1mt-1+(1-b1)g(t).
	        final double beta2           = 0.99; // vt=b2vt-1+(1-b2)g2t.
	        final double epsilon         = 1e-8; // avoids division by 0, smoothing
	        final boolean biascorrection = true; 
	       
	        
	        
	        // Anzeige parameter
	        final int ystretch =3;
	        boolean feedback = true; // Testfall, vorgegebner Input oder eigenen Output als Input nutzen?
	        boolean online = false; 
	        
	        if(Dim==2){
	        	BouncingBall2D bb = new BouncingBall2D(random,in,out,hid,train,trainmlp,epochs,length,trainsize,learningrate,beta1,beta2,epsilon,biascorrection,ystretch,feedback, online);
	        	
	        }else{
	        	BouncingBall1D bb = new BouncingBall1D(random,in,out,hid,train,trainmlp,epochs,length,trainsize,learningrate,beta1,beta2,epsilon,biascorrection,ystretch,feedback, online);
	            
	        }
		
	}
}
