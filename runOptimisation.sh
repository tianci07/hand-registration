python3 HandRegistration.py --target "./00382-s1-neg1.png"\
			    --output "./00382-s1-neg1/MAE-restart"\
			    --restart 0\
			    --parameters 24\
			    --angles 22\
			    --max_mutation_sigma 0.1\
   			    --min_mutation_sigma 0.01\
			    --individuals 20\
			    --generations 20\
			    --elitism 0.1\
			    --new_blood 0.1\
			    --gaussian_mutation 0.1 0.2\
			    --blend_cross_over 0.6\
			    --plot_metrics plot
