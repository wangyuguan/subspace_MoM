function [ w, t ] = circle_rule ( nt )

%*****************************************************************************80
%
%% circle_rule() computes a quadrature rule for the unit circle.
%
%  Discussion:
%
%    The unit circle is the region:
%
%      x * x + y * y = 1.
%
%    The integral I(f) is then approximated by
%
%      Q(f) = 2 * pi * sum ( 1 <= i <= NT ) W(i) * F ( cos(T(i)), sin(T(i)) ).
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    06 April 2014
%
%  Author:
%
%    John Burkardt
%
%  Input:
%
%    integer NT, the number of angles to use.
%
%  Output:
%
%    real W(NT), the weights for the rule.
%
%    real T(NT), the angles for the rule.
%
  w(1:nt,1) = 1.0 / nt;
  t(1:nt,1) = 2.0 * pi * ( 0 : nt - 1 )' / nt;

  return
end
