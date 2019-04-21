#!/usr/bin/ruby
require './tools/lg/lg-hex'
class MoHexBot < BenzeneBot
    print "Enter little golem ID:"
    loginID=gets
    LOGIN=loginID
    BOSS_ID=loginID
    def initialize
        @supported_gametypes = /Hex 13x13/
        print "Enter Password for #{LOGIN}:"
        ps=gets
        #ps="cujopevy"
        ps=ps.chomp
        super(LOGIN,ps,BOSS_ID)
    end
    def genmove(size, moves)
        gtp = GTPClient.new(@logger, "build/src/mohex/mohex3HNN --config=config.htp")
        sleep 1
        gtp.cmd('boardsize ' + size.to_s)
        gtp.cmd('param_game on_little_golem 1')

        #ignore swap-move if opponent has played a swap
        if translate_LG2Hex!(moves)
            gtp.cmd('param_game allow_swap 0')
        end
        self.log("moves played: "+moves.join(','))
        gtp.cmd('play-game ' + moves.join(' '))
        gtp.cmd('showboard')
        gtp.cmd('nn_evaluate')
        response = gtp.cmd('genmove ' + (moves.length % 2 == 0 ? 'b' : 'w'))
        print "generated move:"+response
        gtp.cmd('quit')
        sleep 1
        gtp.close
        return response[2..-3]
    end
end

#enable to cause the http library to show warnings/errors
$VERBOSE = 1
w=MoHexBot.new
loop {
    begin
        while w.parse
        end
        sleep(5)
    rescue Timeout::Error => e
        p 'timeout error (rbuff_fill exception), try again in 30 seconds'
		sleep(30)
    rescue => e
        p e.message
        p e.backtrace
        p 'An error... wait 5 minutes'
        sleep(300)
    end
}
