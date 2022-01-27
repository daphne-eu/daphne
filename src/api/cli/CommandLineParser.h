/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_API_CLI_COMMAND_LINE_PARSER
#define SRC_API_CLI_COMMAND_LINE_PARSER
#include <string>
#include <map>
#include <vector>
#include <utility>

using namespace std;

class CommandLineParser {

    public:
        map<string, string> argsMap;
        map<string, string> argsDefined;

        //vector<string> options;
        //vector<string> descriptions;

        string helpMessage="Usage: build/bin/daphnec FILE [options]\nAvailable options are\n";
        string errorMessage="";
        void init(){
            helpMessage="Usage: build/bin/daphnec FILE [options]\nAvailable options are\n";
            errorMessage="";
            argsMap.clear();
            argsDefined.clear();
        }
    public:
        CommandLineParser(){
        }
        bool addNewOption(string Option, string Description, string PossibleValues){
            //options.push_back(Option);
            //description.push_back(Description);
            if(Option.length()<3 || (Option.at(0)!='-' && Option.at(1)!='-')){
                errorMessage="option has wrong format it should be --<something>";
                return false;
            } 
            //register new option
            //if option exists no need to duplicate, the function takes only the first instance of the option
            if(isExist(Option)){
                errorMessage= "option exists no need to duplicate, the function takes only the first instance of the option";
                return false;
            }


            argsMap.insert({Option, ""});

            // construct the help message
            helpMessage.append(Option);
            helpMessage.append("\t");
            helpMessage.append(Description);
            helpMessage.append("\n");

            return true;
        }
        string getHelpMessage(){
            return helpMessage;
        }
        bool isDefined(string Option){
            return argsDefined.find(Option) != argsDefined.end();
        }
        bool isExist(string Option){
            return argsMap.find(Option) != argsMap.end();
        }
        string getArgValue(string Option){
            return argsDefined[Option];
        }
        string getUsageMessage(string Command){
            string tempMessage= "Usage: ";
            tempMessage.append(Command);
            tempMessage.append(" FILE ");
            tempMessage.append("[options]\n");
            tempMessage.append("for more details ");
            tempMessage.append(Command);
            tempMessage.append(" --help\n");
            return tempMessage;
        }
        string getError(){
            return errorMessage; 
        }
        void clearInternalData(){
            argsDefined.clear();
        }
        void clearError(){
            errorMessage="";
        }
        bool parseCommandLine(int argc, char** argv){
            clearInternalData();
            if(argc<2){
                errorMessage=getUsageMessage(argv[0]); 
                return false;
            }
            string previous="";
            for(int i=2;i<argc;i++)
            {
                string current=argv[i];
                if(current=="")
                    continue;
                if(current.length()<2 ||!(current.at(0)=='-' && current.at(1)=='-')){
                    if(previous==""){
                        errorMessage="unknown error! value "+current+" is not attached to any argument";
                        return false;
                    }
                    else{
                        if(!isExist(previous)){
                            errorMessage="unknown error! check argument list --help for more details "+previous;
                            return false;
                        }   
                        if(isDefined(previous)){
                            errorMessage="duplicated definition of the argument "+ previous;
                            return false;
                        } 
                        cout<<previous<<" has been added "<< "with value "<< current<<endl;
                        argsDefined.insert({previous,current});
                        previous="";
                    }                                
                }
                else{
                    previous=current;
                } 
            }
            if(previous!=""){
                if(!isExist(previous)){
                    errorMessage="unknown error! check argument list --help for more details "+previous;
                    return false;
                }
                if(isDefined(previous)){
                    errorMessage="duplicated definition of the argument "+ previous;
                    return false;
                }
                argsDefined.insert({previous,""});
            }
            return true;
        }
};
#endif
