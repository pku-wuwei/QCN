#!/usr/bin/perl
#
#  Author: Preslav Nakov and Walid Magdy
#  
#  Description: Scores subtask A of SemEval-2015 Task 3.
#
#  Version: 2.0
#
#  Last modified: November 12, 2014
#
#
#  Use:
#     SemEval2015-task3-scorer-subtaskA.pl <GOLD_FILE> <PREDICTIONS_FILE>
#
#  Example use:
#     SemEval2015-task3-scorer-subtaskA.pl CQA-QL-devel-gold.txt CQA-QL-devel-predicted.txt > SemEval2015-task3-scores-subtaskA.txt
#     SemEval2015-task3-scorer-subtaskA.pl CQA-QL-devel-gold.txt CQA-QL-devel-predicted-majority-baseline.txt > SemEval2015-task3-scores-subtaskA-majority-baseline.txt
#
#  Description:
#     The scorer takes as input a proposed classification file and an answer key file.
#     Both files should contain one prediction per line in the format "<Comment_ID>	<Label>"
#     with a TAB as a separator, e.g.,
#           Q2601_C1	Dialogue
#			Q2601_C2	Good
#			Q2601_C3	Potential
#               ...
#     The files have to be sorted in the same way, i.e., their <Comment_ID> should match for each line!
#     Repetitions of IDs are not allowed in either of the files.
#
#     The scorer calculates and outputs the following statistics:
#        (1) confusion matrix, which shows
#			- the count for each gold/predicted pair
#           - the sums for each row/column: -SUM-
#        (2) accuracy
#        (3) precision (P), recall (R), and F1-score for each label
#        (4) micro-averaged P, R, F1
#        (5) macro-averaged P, R, F1
#
#     The scoring is done two times:
#       (i)   using fine-grained labels  : Good, Bad, Potential, Dialogue (where Bad also includes Not English and Other)
#       (ii)  using coarse-grained labels: Good, Bad, Potential (where Bad also includes Dialogue, Not English and Other)
#     
#     The official score is the macro-averaged F1-score for (ii).
#
#	  ###########################################
#	  Version 2.0 is a modification of version one to accommodate the Arabic task as well
#	  ###########################################

use warnings;
use strict;
use utf8;


###################
###   GLOBALS   ###
###################

my %confMatrixCoarse   = ();
my @allLabelsCoarse    = ('Good', 'Bad', 'Pot.');
my %labelMappingCoarse = ('Good'=>'Good', 'Bad'=>'Bad', 'Potential'=>'Pot.', 'Dialogue'=>'Bad', 'Not English'=>'Bad', 'Other'=>'Bad', 'direct'=>'Good', 'related'=>'Pot.', 'irrelevant'=>'Bad');

my %confMatrixFine     = ();
my @allLabelsFine      = ('Good', 'Bad', 'Pot.', 'Dial');
my %labelMappingFine   = ('Good'=>'Good', 'Bad'=>'Bad', 'Potential'=>'Pot.', 'Dialogue'=>'Dial', 'Not English'=>'Bad', 'Other'=>'Bad', 'direct'=>'Good', 'related'=>'Pot.', 'irrelevant'=>'Bad');


################
###   MAIN   ###
################

### 1. Check parameters
die "Usage: $0 <GOLD_FILE> <PREDICTIONS_FILE>\n" if ($#ARGV != 1);
my $GOLD_FILE        = $ARGV[0];
my $PREDICTIONS_FILE = $ARGV[1];

### 2. Open the files
open GOLD, $GOLD_FILE or die "Error opening $GOLD_FILE!";
open PREDICTED, $PREDICTIONS_FILE or die "Error opening $PREDICTIONS_FILE!";
binmode(GOLD, ":utf8");
binmode(PREDICTED, ":utf8");

### 3. Collect the statistics
my %CIDs = ();
while (<GOLD>) {
	
	# 3.1. Get the GOLD CID and label
	# Q2601_C1	Dialogue
	die "Wrong file format!" if (!/^Q*([0-9]+[_-]C*[0-9]+)\t(Good|Bad|Potential|Dialogue|Not English|Other|direct|related|irrelevant)[\n\r]*$/);
	my ($goldCID, $goldLabel) = ($1, $2);

	# 3.2. Get the PREDICTED CID and label
	# Q2601_C1	Bad
	die "The file $PREDICTIONS_FILE is shorter!" if (!($_ = <PREDICTED>));
	die "Wrong file format!" if (!/^Q*([0-9]+[_-]C*[0-9]+)\t(Good|Bad|Potential|Dialogue|Not English|Other|direct|related|irrelevant)[\n\r]*$/);
	my ($predictedCID, $predictedLabel) = ($1, $2);

	# 3.3. Make sure IDs match
	die "Comment IDs differ: gold='$goldCID' predicted='$predictedCID'" if ($predictedCID ne $goldCID);

	# 3.4. Make sure this Comment ID was not seen already
	die "Repetition of Comment ID: $goldCID\n" if (defined $CIDs{$goldCID});
	$CIDs{$goldCID}++;

	# 3.5. Update the statistics
	$confMatrixFine{$labelMappingFine{$predictedLabel}}{$labelMappingFine{$goldLabel}}++;
	$confMatrixCoarse{$labelMappingCoarse{$predictedLabel}}{$labelMappingCoarse{$goldLabel}}++;
}

### 4. Fine-grained evaluation
print "\n<<< I. FINE-GRAINED EVALUATION >>>\n\n";
&evaluate(\@allLabelsFine, \%confMatrixFine);

### 5. Coarse-grained evaluation
print "\n<<< II. COARSE EVALUATION >>>\n\n";
my $officialScore = &evaluate(\@allLabelsCoarse, \%confMatrixCoarse);

### 6. Output the official score
print "\n<<< III. OFFICIAL SCORE >>>\n";
printf "\nMACRO-averaged coarse-grained F1: %6.2f%s", $officialScore, "%\n";

### 7. Close the files
close GOLD or die;
close PREDICTED or die;


################
###   SUBS   ###
################

sub evaluate() {
	my ($allLabels, $confMatrix) = @_;

	### 0. Calculate the horizontal and vertical sums
	my %allLabelsProposed = ();
	my %allLabelsAnswer   = ();
	my ($cntCorrect, $cntTotal) = (0, 0);
	foreach my $labelGold (@{$allLabels}) {
		foreach my $labelProposed (@{$allLabels}) {
			$$confMatrix{$labelProposed}{$labelGold} = 0
				if (!defined($$confMatrix{$labelProposed}{$labelGold}));
			$allLabelsProposed{$labelProposed} += $$confMatrix{$labelProposed}{$labelGold};
			$allLabelsAnswer{$labelGold} += $$confMatrix{$labelProposed}{$labelGold};
			$cntTotal += $$confMatrix{$labelProposed}{$labelGold};
		}
		$cntCorrect += $$confMatrix{$labelGold}{$labelGold};
	}

	### 1. Print the confusion matrix heading
	print "Confusion matrix:\n";
	print "       ";
	foreach my $label (@{$allLabels}) {
		printf " %4s", $label;
	}
	print " <-- classified as\n";
	print "      +";
	foreach (@{$allLabels}) {
		print "-----";
	}
	print "+ -SUM-\n";

	### 2. Print the rest of the confusion matrix
	my $freqCorrect = 0;
	foreach my $labelGold (@{$allLabels}) {

		### 2.1. Output the short relation label
		printf " %4s |", $labelGold;

		### 2.2. Output a row of the confusion matrix
		foreach my $labelProposed (@{$allLabels}) {
			printf "%4d ", $$confMatrix{$labelProposed}{$labelGold};
		}

		### 2.3. Output the horizontal sums
		printf "| %4d\n", $allLabelsAnswer{$labelGold};
	}
	print "      +";
	foreach (@{$allLabels}) {
		print "-----";
	}
	print "+\n";
	
	### 3. Print the vertical sums
	print " -SUM- ";
	foreach my $labelProposed (@{$allLabels}) {
		printf "%4d ", $allLabelsProposed{$labelProposed};
	}
	print "\n\n";

	### 5. Output the accuracy
	my $accuracy = 100.0 * $cntCorrect / $cntTotal;
	printf "%s%d%s%d%s%5.2f%s", 'Accuracy (calculated for the above confusion matrix) = ', $cntCorrect, '/', $cntTotal, ' = ', $accuracy, "\%\n";

	### 8. Output P, R, F1 for each relation
	my ($macroP, $macroR, $macroF1) = (0, 0, 0);
	my ($microCorrect, $microProposed, $microAnswer) = (0, 0, 0);
	print "\nResults for the individual labels:\n";
	foreach my $labelGold (@{$allLabels}) {

		### 8.3. Calculate P/R/F1
		my $P  = (0 == $allLabelsProposed{$labelGold}) ? 0
				: 100.0 * $$confMatrix{$labelGold}{$labelGold} / $allLabelsProposed{$labelGold};
		my $R  = (0 == $allLabelsAnswer{$labelGold}) ? 0
				: 100.0 * $$confMatrix{$labelGold}{$labelGold} / $allLabelsAnswer{$labelGold};
		my $F1 = (0 == $P + $R) ? 0 : 2 * $P * $R / ($P + $R);

		printf "%10s%s%4d%s%4d%s%6.2f", $labelGold,
			" :    P = ", $$confMatrix{$labelGold}{$labelGold}, '/', $allLabelsProposed{$labelGold}, ' = ', $P;

		printf"%s%4d%s%4d%s%6.2f%s%6.2f%s\n", 
		  	 "%     R = ", $$confMatrix{$labelGold}{$labelGold}, '/', $allLabelsAnswer{$labelGold},   ' = ', $R,
			 "%     F1 = ", $F1, '%';

		### 8.5. Accumulate statistics for micro/macro-averaging
		if ($labelGold ne '_Other') {
			$macroP  += $P;
			$macroR  += $R;
			$macroF1 += $F1;
			$microCorrect += $$confMatrix{$labelGold}{$labelGold};
			$microProposed += $allLabelsProposed{$labelGold};
			$microAnswer += $allLabelsAnswer{$labelGold};
		}
	}

	### 9. Output the micro-averaged P, R, F1
	my $microP  = (0 == $microProposed)    ? 0 : 100.0 * $microCorrect / $microProposed;
	my $microR  = (0 == $microAnswer)      ? 0 : 100.0 * $microCorrect / $microAnswer;
	my $microF1 = (0 == $microP + $microR) ? 0 :   2.0 * $microP * $microR / ($microP + $microR);
	print "\nMicro-averaged result:\n";
	printf "%s%4d%s%4d%s%6.2f%s%4d%s%4d%s%6.2f%s%6.2f%s\n",
		      "P = ", $microCorrect, '/', $microProposed, ' = ', $microP,
		"%     R = ", $microCorrect, '/', $microAnswer, ' = ', $microR,
		"%     F1 = ", $microF1, '%';

	### 10. Output the macro-averaged P, R, F1
	my $distinctLabelsCnt = $#{$allLabels}+1; 

	$macroP  /= $distinctLabelsCnt; # first divide by the number of non-Other categories
	$macroR  /= $distinctLabelsCnt;
	$macroF1 /= $distinctLabelsCnt;
	print "\nMACRO-averaged result:\n";
	printf "%s%6.2f%s%6.2f%s%6.2f%s\n\n\n\n", "P = ", $macroP, "%\tR = ", $macroR, "%\tF1 = ", $macroF1, '%';

	### 11. Return the official score
	return $macroF1;
}
