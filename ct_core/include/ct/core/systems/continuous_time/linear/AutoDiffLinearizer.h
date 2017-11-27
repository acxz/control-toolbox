/**********************************************************************************************************************
This file is part of the Control Toolbox (https://adrlab.bitbucket.io/ct), copyright by ETH Zurich, Google Inc.
Authors:  Michael Neunert, Markus Giftthaler, Markus Stäuble, Diego Pardo, Farbod Farshidian
Licensed under Apache2 license (see LICENSE file in main directory)
**********************************************************************************************************************/

#pragma once

namespace ct {
namespace core {

//! Computes the linearization of a general non-linear ControlledSystem using Automatic Differentiation (without code generation)
/*!
 * This class takes a non-linear ControlledSystem \f$ \dot{x} = f(x,u,t) \f$ and computes the linearization
 * around a certain point \f$ x = x_s \f$, \f$ u = u_s \f$.
 *
 * \f[
 *   \dot{x} = A x + B u
 * \f]
 *
 * where
 *
 * \f[
 * \begin{aligned}
 * A &= \frac{df}{dx} |_{x=x_s, u=u_s} \\
 * B &= \frac{df}{du} |_{x=x_s, u=u_s}
 * \end{aligned}
 * \f]
 *
 * \note This is generally the most accurate way to generate the linearization of system dynamics together with ADCodegenLinearizer.
 * However, the latter is much faster. Consider using the latter for production code.
 *
 * Unit test \ref AutoDiffLinearizerTest.cpp illustrates the use of the AutoDiffLinearizer.
 *
 *
 * \warning You should ensure that your ControlledSystem is templated on the scalar type and does not contain branching
 * (if/else statements, switch cases etc.)
 *
 *
 * \warning This function still has some issues with pure time dependency
 * \todo Make time an Auto-Diff parameter
 *
 * @tparam dimension of state vector
 * @tparam dimension of control vector
 */
template <size_t STATE_DIM, size_t CONTROL_DIM>
class AutoDiffLinearizer : public LinearSystem<STATE_DIM, CONTROL_DIM>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    typedef CppAD::AD<double> AD_double;                                   //!< Auto-Diff double type
    typedef LinearSystem<STATE_DIM, CONTROL_DIM> Base;                     //!< Base class type

    typedef StateVector<STATE_DIM, double> state_vector_t;                 //!< state vector type
    typedef ControlVector<CONTROL_DIM, double> control_vector_t;           //!< input vector type
    typedef typename Base::state_matrix_t state_matrix_t;                  //!< state Jacobian type
    typedef typename Base::state_control_matrix_t state_control_matrix_t;  //!< input Jacobian type

    typedef ControlledSystem<STATE_DIM, CONTROL_DIM, AD_double> system_t;     //!< type of system to be linearized

    //! default constructor
    /*!
	 * @param nonlinearSystem non-linear system instance to linearize
	 */
    AutoDiffLinearizer(std::shared_ptr<system_t> nonlinearSystem)
        : Base(nonlinearSystem->getType()),
          nonlinearSystem_(nonlinearSystem),
          linearizer_(std::bind(&system_t::computeControlledDynamics, nonlinearSystem_.get(), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4))
    {
    }

    //! copy constructor
    AutoDiffLinearizer(const AutoDiffLinearizer& arg)
        : Base(arg.nonlinearSystem_->getType()),
          nonlinearSystem_(arg.nonlinearSystem_->clone()),
          linearizer_(std::bind(&system_t::computeControlledDynamics, nonlinearSystem_.get(), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)),
          dFdx_(arg.dFdx_),
          dFdu_(arg.dFdu_)
    {
    }

    //! destructor
    virtual ~AutoDiffLinearizer() {}

    //! deep cloning
    AutoDiffLinearizer<STATE_DIM, CONTROL_DIM>* clone() const override
    {
        return new AutoDiffLinearizer<STATE_DIM, CONTROL_DIM>(*this);
    }

    //! get the Jacobian with respect to the state
    /*!
	 * This computes the linearization of the system with respect to the state at a given point \f$ x=x_s \f$, \f$ u=u_s \f$,
	 * i.e. it computes
	 *
	 * \f[
	 * A = \frac{df}{dx} |_{x=x_s, u=u_s}
	 * \f]
	 *
	 * @param x state to linearize at
	 * @param u control to linearize at
	 * @param t time
	 * @return Jacobian wrt state
	 */
    virtual const state_matrix_t& getDerivativeState(const state_vector_t& x,
        const control_vector_t& u,
        const double t = 0.0) override
    {
        dFdx_ = linearizer_.getDerivativeState(x, u, t);
        return dFdx_;
    }


    //! get the Jacobian with respect to the input
    /*!
	 * This computes the linearization of the system with respect to the input at a given point \f$ x=x_s \f$, \f$ u=u_s \f$,
	 * i.e. it computes
	 *
	 * \f[
	 * B = \frac{df}{du} |_{x=x_s, u=u_s}
	 * \f]
	 *
	 * @param x state to linearize at
	 * @param u control to linearize at
	 * @param t time
	 * @return Jacobian wrt input
	 */
    virtual const state_control_matrix_t& getDerivativeControl(const state_vector_t& x,
        const control_vector_t& u,
        const double t = 0.0) override
    {
        dFdu_ = linearizer_.getDerivativeControl(x, u, t);
        return dFdu_;
    }


protected:
    std::shared_ptr<system_t> nonlinearSystem_;  //!< instance of non-linear system

    DynamicsLinearizerAD<STATE_DIM, CONTROL_DIM, AD_double, double> linearizer_; //!< instance of ad-linearizer

    state_matrix_t dFdx_;
    state_control_matrix_t dFdu_;
};

} // core
} // ct
