#!/usr/bin/ruby
require 'net/http'
require 'yaml'

class String
    def red
        "\e[31m#{self}\e[0m"
    end
    def red_back
        "\e[41m#{self}\e[0m"
    end
    def green
        "\e[32m#{self}\e[0m"
    end  
    def blue
    "\e[34m#{self}\e[0m"
    end
    def yellow
        "\e[33m#{self}\e[0m"
    end
end

class Logger
    def initialize(filename)
        @file = File.new(filename, 'a')
        log '========== Startup =========='
    end
    def timestamp
        return '[' + Time::now.strftime('%y-%m-%d %H:%M:%S') + ']'
    end
    def log(msg)
        puts (timestamp().yellow) + ' ' + msg
        @file.puts timestamp() + ' ' + msg
        @file.flush
    end
    def log_nostamp(msg)
        puts msg
        @file.puts msg
    end
end

class LittleGolemInterface
    def initialize (loginname,psw,boss_id)
        @login,@psw,@boss_id=loginname,psw,boss_id
        @http = Net::HTTP.new('www.littlegolem.net')
        @config_data = {}
        #@logger=Logger.new(loginname + '.log')
        @logger=Logger.new(loginname.sub!(/\s+/, '') + '.log')
    end
    def get_game(gid)
        path="/servlet/sgf/#{gid}/game#{gid}.hgf"
        resp = @http.get2(path, @headers)
        return (resp.code == '200' ? resp.body : nil)
    end
    def get_invitations
        path='/jsp/invitation/index.jsp'
        resp = @http.get2(path, @headers)
        return (resp.code == '200' ? resp.body : nil)
    end
    def send_message(pid,title,msg)
        path="/jsp/message/new.jsp"
        resp = @http.post(path,"messagetitle=#{title}&message=#{msg}&plto=#{pid}", @headers)
        return (resp.code == '200' ? resp.body : nil)
    end
    def post_move(gid,mv,chat = '')
        chat.sub!('+',' leads with ')
        path="/jsp/game/game.jsp?sendgame=#{gid}&sendmove=#{mv}"
        resp = @http.post(path, "message=#{chat}", @headers)
         if resp.code!='200'
            logout
            login
            resp = @http.post(path, "message=#{chat}", @headers)
        end
        return (resp.code == '200' ? resp.body : nil)
    end
    def reply_invitation(inv_id,answer)
        path="/ng/a/Invitation.action?#{answer}=&invid=#{inv_id}"
        resp = @http.get2(path, @headers)
        return (resp.code == '200' ? resp.body : nil)
    end
    def log(msg)
        @logger.log(msg)
    end
    def log_nostamp(msg)
        @logger.log_nostamp(msg)
    end
    def logout
        path="/jsp/login/logoff.jsp"
        resp = @http.get2(path, @headers)
        @headers = nil
        return (resp.code == '200' ? resp.body : nil)
    end
    def login
        path='/jsp/login/index.jsp'
	    resp = @http.get2(path, nil)
        @headers = {'Cookie' => resp['set-cookie'] }
        data = "login=#{@login}&password=#{@psw}"
        resp = @http.post(path, data, @headers)
        return (resp.code == '200' ? resp.body : nil)
    end
    def get_gamesheet
        path='/jsp/game/index.jsp'
        resp = @http.get2(path, @headers)
        return (resp.code == '200' ? resp.body : nil)
    end
    def parse
        if !self.login
            self.log('login failed')
	        print "login failed....\n"
            return false;
        end
        if (gamesheet = get_gamesheet)
            if gamesheet =~ /You have [1-9][0-9]* invitations to game/
		        print "new invitations\n"
                invites = get_invitations
                if (invites) 
                    a = invites.slice(/Invitations.*Your decision.*?table>/m).scan(/<td>(Hex.*Size.*)<\/td>/m).flatten
                    b=a[0].scan(/1[3|9]/)
                    invid_id=a[0].scan(/invid=[0-9]+/)[0].scan(/[0-9]+/)
                    opponent = a[1]
                    gametype = b[0]
                    if b[0].to_i == 13 or b[0].to_i == 19
                        self.log("supported board size "+ b[0]+". Going to accpet.\n")
                        answer='accept'
                    else
                        self.log("unsupported board size " + b[0] +". Refuse.\n")
                        answer='refuse'
                    end
                    #self.send_message(@boss_id,"New invitation","#{answer} #{gametype} from #{opponent}")
                    self.log("#{answer} '#{gametype}' from #{opponent}")
                    #inv_id = a[5].scan(/invid=(\d*)?/m)[0]
                    invid_id=a[0].scan(/invid=[0-9]+/)[0].scan(/[0-9]+/)[0]
                    print "invid_id:"
                    p invid_id
                    reply_invitation(invid_id, answer)
                end
            end
            
            if (gamesheet =~  /You have [1-9][0-9]* games on move/)
                gameids=gamesheet.slice(/games on move.*See all games/m).scan(/gid=(\d+)?.*/).flatten
                print "gameids:\n"
                puts gameids
                parse_make_moves(gameids)
                return true;
            else
                self.log("No games found where it's my turn. Sleeping...")
                return false
            end
        end
    end
end
